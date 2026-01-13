import * as THREE from 'three';
import { Reflector } from './utils/Reflector.js';
import { PolicyRunner } from './policyRunner.js';
import { toFloatArray } from './utils/math.js';

const MOTION_INDEX_FORMAT = 'tracking-motion-index-v1';

function stripJsonExtension(path) {
  const file = path.split('/').pop() ?? path;
  return file.replace(/\.json$/i, '');
}

function normalizeMotionEntry(entry) {
  if (typeof entry === 'string') {
    return { name: stripJsonExtension(entry), file: entry };
  }
  if (entry && typeof entry === 'object') {
    const file = entry.file ?? entry.path ?? null;
    if (!file) {
      return null;
    }
    const name = entry.name ?? stripJsonExtension(file);
    return { name, file };
  }
  return null;
}

function parseMotionIndex(payload) {
  if (!payload || typeof payload !== 'object') {
    return null;
  }
  if (payload.format !== MOTION_INDEX_FORMAT) {
    return null;
  }
  const motions = Array.isArray(payload.motions) ? payload.motions : [];
  return {
    basePath: payload.base_path ?? null,
    motions
  };
}

async function loadMotionIndex(indexPayload, motionsUrl) {
  const index = parseMotionIndex(indexPayload);
  if (!index) {
    return null;
  }

  const basePath = index.basePath
    ? (index.basePath.endsWith('/') ? index.basePath : `${index.basePath}/`)
    : null;
  const baseUrl = basePath
    ? new URL(basePath, motionsUrl)
    : new URL('.', motionsUrl);
  const motions = {};
  const entries = index.motions.map((entry) => normalizeMotionEntry(entry));

  const requests = entries.map(async (entry) => {
    if (!entry || !entry.file || !entry.name) {
      throw new Error('Motion index entries must include a name and file path.');
    }
    const clipUrl = new URL(entry.file, baseUrl).toString();
    const response = await fetch(clipUrl);
    if (!response.ok) {
      throw new Error(`Failed to load motion clip from ${clipUrl}: ${response.status}`);
    }
    const clip = await response.json();
    motions[entry.name] = clip;
  });

  await Promise.all(requests);
  return motions;
}

export async function reloadScene(mjcf_path) {
  this.scene.remove(this.scene.getObjectByName('MuJoCo Root'));
  const mujoco = this.mujoco;
  console.log('Loading scene:', mjcf_path);

  [this.model, this.data, this.simulation, this.bodies, this.lights] =
    await loadSceneFromURL(mujoco, mjcf_path, this);

  const textDecoder = new TextDecoder();
  const namesArray = new Uint8Array(this.model.names);

  this.jointNamesMJC = [];
  for (let j = 0; j < this.model.njnt; j++) {
    let start_idx = this.model.name_jntadr[j];
    let end_idx = start_idx;
    while (end_idx < namesArray.length && namesArray[end_idx] !== 0) {
      end_idx++;
    }
    this.jointNamesMJC.push(textDecoder.decode(namesArray.subarray(start_idx, end_idx)));
  }

  this.defaultJposPolicy = null;

  this.policyJointNames = null;
  this.ctrl_adr_policy = [];
  this.qpos_adr_policy = [];
  this.qvel_adr_policy = [];
  this.numActions = 0;

  this.timestep = this.model.opt.timestep;
  this.decimation = Math.max(1, Math.round(0.02 / this.timestep));
}

export async function reloadPolicy(policy_path, options = {}) {
  this.currentPolicyPath = policy_path;
  console.log('Reloading policy:', policy_path);

  while (this.policyRunner?.isInferencing) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  const response = await fetch(policy_path);
  if (!response.ok) {
    throw new Error(`Failed to load policy config from ${policy_path}: ${response.status}`);
  }
  const config = await response.json();
  if (options?.onnxPath) {
    config.onnx = { ...(config.onnx ?? {}), path: options.onnxPath };
  }

  let trackingConfig = null;
  if (config.tracking) {
    trackingConfig = { ...config.tracking };
    if (trackingConfig.motions_path && !trackingConfig.motions) {
      const motionsUrl = new URL(trackingConfig.motions_path, window.location.href);
      const response = await fetch(motionsUrl);
      if (!response.ok) {
        throw new Error(`Failed to load tracking motions from ${motionsUrl}: ${response.status}`);
      }
      const payload = await response.json();
      const indexedMotions = await loadMotionIndex(payload, motionsUrl);
      trackingConfig.motions = indexedMotions ?? payload;
    }
  }

  const policyJointNames = Array.isArray(config.policy_joint_names)
    ? config.policy_joint_names
    : null;
  if (!policyJointNames || policyJointNames.length === 0) {
    throw new Error('Policy configuration must include a non-empty policy_joint_names list');
  }

  configureJointMappings(this, policyJointNames);
  const configDefaultJointPos = Array.isArray(config.default_joint_pos)
    ? config.default_joint_pos
    : null;
  if (configDefaultJointPos) {
    if (configDefaultJointPos.length !== this.numActions) {
      throw new Error(
        `default_joint_pos length ${configDefaultJointPos.length} does not match policy_joint_names length ${this.numActions}`
      );
    }
    this.defaultJposPolicy = new Float32Array(configDefaultJointPos);
  }
  this.kpPolicy = toFloatArray(config.stiffness, this.numActions, 0.0);
  this.kdPolicy = toFloatArray(config.damping, this.numActions, 0.0);
  this.control_type = config.control_type ?? 'joint_position';

  if (trackingConfig) {
    trackingConfig.policy_joint_names = policyJointNames.slice();
  }

  this.simulation.resetData();
  this.simulation.forward();
  this.policyRunner = new PolicyRunner(
    {
      ...config,
      tracking: trackingConfig,
      policy_joint_names: policyJointNames,
      action_scale: config.action_scale,
      default_joint_pos: this.defaultJposPolicy
    },
    {
      policyJointNames,
      actionScale: config.action_scale,
      defaultJointPos: this.defaultJposPolicy
    }
  );
  await this.policyRunner.init();

  const state = this.readPolicyState?.();
  if (state) {
    this.policyRunner.reset(state);
  } else {
    this.policyRunner.reset();
  }

  this.params.current_motion = 'default';
}

export async function loadSceneFromURL(mujoco, filename, parent) {
  if (parent.simulation) {
    parent.simulation.free();
    parent.simulation = null;
    parent.model = null;
    parent.data = null;
  }

  const model = mujoco.MjModel.loadFromXML('/working/' + filename);
  const data = new mujoco.MjData(model);
  const simulation = createSimulationWrapper(mujoco, model, data);

  const textDecoder = new TextDecoder('utf-8');
  const names_array = new Uint8Array(model.names);
  const fullString = textDecoder.decode(model.names);
  const names = fullString.split(textDecoder.decode(new ArrayBuffer(1)));

  const mujocoRoot = new THREE.Group();
  mujocoRoot.name = 'MuJoCo Root';
  parent.scene.add(mujocoRoot);

  const bodies = {};
  const meshes = {};
  const lights = [];

  let material = new THREE.MeshPhysicalMaterial();
  material.color = new THREE.Color(1, 1, 1);

  for (let g = 0; g < model.ngeom; g++) {
    if (!(model.geom_group[g] < 3)) { continue; }

    let b = model.geom_bodyid[g];
    let type = model.geom_type[g];
    let size = [
      model.geom_size[(g * 3) + 0],
      model.geom_size[(g * 3) + 1],
      model.geom_size[(g * 3) + 2]
    ];

    if (!(b in bodies)) {
      bodies[b] = new THREE.Group();

      let start_idx = model.name_bodyadr[b];
      let end_idx = start_idx;
      while (end_idx < names_array.length && names_array[end_idx] !== 0) {
        end_idx++;
      }
      let name_buffer = names_array.subarray(start_idx, end_idx);
      bodies[b].name = textDecoder.decode(name_buffer);

      bodies[b].bodyID = b;
      bodies[b].has_custom_mesh = false;

      if (bodies[b].name === 'base') {
        parent.pelvis_body_id = b;
      }
    }

    let geometry = new THREE.SphereGeometry(size[0] * 0.5);
    if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) {
      // Special handling for plane later.
    } else if (type == mujoco.mjtGeom.mjGEOM_HFIELD.value) {
      // TODO: Implement this.
    } else if (type == mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(size[0]);
    } else if (type == mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
    } else if (type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1);
    } else if (type == mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
    } else if (type == mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(size[0] * 2.0, size[2] * 2.0, size[1] * 2.0);
    } else if (type == mujoco.mjtGeom.mjGEOM_MESH.value) {
      let meshID = model.geom_dataid[g];

      if (!(meshID in meshes)) {
        geometry = new THREE.BufferGeometry();

        let vertex_buffer = model.mesh_vert.subarray(
          model.mesh_vertadr[meshID] * 3,
          (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3);
        for (let v = 0; v < vertex_buffer.length; v += 3) {
          let temp = vertex_buffer[v + 1];
          vertex_buffer[v + 1] = vertex_buffer[v + 2];
          vertex_buffer[v + 2] = -temp;
        }

        let normal_buffer = model.mesh_normal.subarray(
          model.mesh_vertadr[meshID] * 3,
          (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3);
        for (let v = 0; v < normal_buffer.length; v += 3) {
          let temp = normal_buffer[v + 1];
          normal_buffer[v + 1] = normal_buffer[v + 2];
          normal_buffer[v + 2] = -temp;
        }

        let uv_buffer = model.mesh_texcoord.subarray(
          model.mesh_texcoordadr[meshID] * 2,
          (model.mesh_texcoordadr[meshID] + model.mesh_vertnum[meshID]) * 2);
        let triangle_buffer = model.mesh_face.subarray(
          model.mesh_faceadr[meshID] * 3,
          (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(vertex_buffer, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(normal_buffer, 3));
        geometry.setAttribute('uv', new THREE.BufferAttribute(uv_buffer, 2));
        geometry.setIndex(Array.from(triangle_buffer));
        meshes[meshID] = geometry;
      } else {
        geometry = meshes[meshID];
      }

      bodies[b].has_custom_mesh = true;
    }

    let texture = undefined;
    let color = [
      model.geom_rgba[(g * 4) + 0],
      model.geom_rgba[(g * 4) + 1],
      model.geom_rgba[(g * 4) + 2],
      model.geom_rgba[(g * 4) + 3]
    ];
    if (model.geom_matid[g] != -1) {
      let matId = model.geom_matid[g];
      color = [
        model.mat_rgba[(matId * 4) + 0],
        model.mat_rgba[(matId * 4) + 1],
        model.mat_rgba[(matId * 4) + 2],
        model.mat_rgba[(matId * 4) + 3]
      ];

      texture = undefined;
      const mjNTEXROLE = 10;
      const mjTEXROLE_RGB = 1;
      let texId = model.mat_texid[(matId * mjNTEXROLE) + mjTEXROLE_RGB];

      if (texId != -1) {
        let width = model.tex_width[texId];
        let height = model.tex_height[texId];
        let offset = model.tex_adr[texId];
        let channels = model.tex_nchannel[texId];
        let texData = model.tex_data;
        let rgbaArray = new Uint8Array(width * height * 4);
        for (let p = 0; p < width * height; p++) {
          rgbaArray[(p * 4) + 0] = texData[offset + ((p * channels) + 0)];
          rgbaArray[(p * 4) + 1] = channels > 1 ? texData[offset + ((p * channels) + 1)] : rgbaArray[(p * 4) + 0];
          rgbaArray[(p * 4) + 2] = channels > 2 ? texData[offset + ((p * channels) + 2)] : rgbaArray[(p * 4) + 0];
          rgbaArray[(p * 4) + 3] = channels > 3 ? texData[offset + ((p * channels) + 3)] : 255;
        }
        texture = new THREE.DataTexture(rgbaArray, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
        if (texId == 2) {
          texture.repeat = new THREE.Vector2(100, 100);
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
        } else {
          texture.repeat = new THREE.Vector2(
            model.mat_texrepeat[(model.geom_matid[g] * 2) + 0],
            model.mat_texrepeat[(model.geom_matid[g] * 2) + 1]);
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
        }

        texture.needsUpdate = true;
      }
    }

    const materialOptions = {
      color: new THREE.Color(color[0], color[1], color[2]),
      transparent: color[3] < 1.0,
      opacity: color[3] / 255,
      specularIntensity: model.geom_matid[g] != -1 ? model.mat_specular[model.geom_matid[g]] : undefined,
      reflectivity: model.geom_matid[g] != -1 ? model.mat_reflectance[model.geom_matid[g]] : undefined,
      roughness: model.geom_matid[g] != -1 ? 1.0 - model.mat_shininess[model.geom_matid[g]] * -1.0 : undefined,
      metalness: model.geom_matid[g] != -1 ? model.mat_metallic[model.geom_matid[g]] : undefined
    };
    if (texture) {
      materialOptions.map = texture;
    }

    let currentMaterial = new THREE.MeshPhysicalMaterial(materialOptions);

    let mesh = new THREE.Mesh();
    if (type == 0) {
      mesh = new Reflector(new THREE.PlaneGeometry(100, 100), { clipBias: 0.003, texture });
      mesh.rotateX(-Math.PI / 2);
      mesh.material.depthWrite = false;
      mesh.renderOrder = -1;
    } else {
      mesh = new THREE.Mesh(geometry, currentMaterial);
    }

    mesh.castShadow = g == 0 ? false : true;
    mesh.receiveShadow = type != 7;
    mesh.bodyID = b;
    bodies[b].add(mesh);
    getPosition(model.geom_pos, g, mesh.position);
    if (type != 0) {
      getQuaternion(model.geom_quat, g, mesh.quaternion);
    }
    if (type == 4) {
      mesh.scale.set(size[0], size[2], size[1]);
    }
  }

  let tendonMat = new THREE.MeshPhongMaterial();
  tendonMat.color = new THREE.Color(0.8, 0.3, 0.3);
  mujocoRoot.cylinders = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(1, 1, 1),
    tendonMat,
    1023
  );
  mujocoRoot.cylinders.receiveShadow = true;
  mujocoRoot.cylinders.castShadow = true;
  mujocoRoot.add(mujocoRoot.cylinders);
  mujocoRoot.spheres = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 10, 10),
    tendonMat,
    1023
  );
  mujocoRoot.spheres.receiveShadow = true;
  mujocoRoot.spheres.castShadow = true;
  mujocoRoot.add(mujocoRoot.spheres);

  for (let l = 0; l < model.nlight; l++) {
    let light = new THREE.SpotLight();
    if (model.light_type[l] == 0) {
      light = new THREE.SpotLight();
    } else if (model.light_type[l] == 1) {
      light = new THREE.DirectionalLight();
    } else if (model.light_type[l] == 2) {
      light = new THREE.PointLight();
    } else if (model.light_type[l] == 3) {
      light = new THREE.HemisphereLight();
    }
    if (model.light_diffuse && model.light_diffuse.length >= (l + 1) * 3) {
      const r = model.light_diffuse[(l * 3) + 0];
      const g = model.light_diffuse[(l * 3) + 1];
      const b = model.light_diffuse[(l * 3) + 2];
      const max = Math.max(r, g, b);
      if (max > 0) {
        light.color.setRGB(r / max, g / max, b / max);
        light.intensity = max;
      } else {
        light.color.setRGB(1, 1, 1);
        light.intensity = 1;
      }
    }
    light.decay = model.light_attenuation[l] * 100;
    light.penumbra = 0.5;
    if (model.light_castshadow) {
      light.castShadow = model.light_castshadow[l] !== 0;
    } else {
      light.castShadow = true;
    }

    light.shadow.mapSize.width = 1024;
    light.shadow.mapSize.height = 1024;
    light.shadow.camera.near = 0.1;
    light.shadow.camera.far = 10;
    if (bodies[0]) {
      bodies[0].add(light);
    } else {
      mujocoRoot.add(light);
    }
    lights.push(light);
  }
  if (model.nlight == 0) {
    let light = new THREE.DirectionalLight();
    mujocoRoot.add(light);
  }

  for (let b = 0; b < model.nbody; b++) {
    if (b == 0 || !bodies[0]) {
      mujocoRoot.add(bodies[b]);
    } else if (bodies[b]) {
      bodies[0].add(bodies[b]);
    } else {
      bodies[b] = new THREE.Group();
      bodies[b].name = names[b + 1];
      bodies[b].bodyID = b;
      bodies[b].has_custom_mesh = false;
      bodies[0].add(bodies[b]);
    }
  }

  parent.mujocoRoot = mujocoRoot;
  if (parent.lastSimState) {
    parent.lastSimState.bodies = new Map();
    parent.lastSimState.lights = new Map();
    parent.lastSimState.tendons = {
      numWraps: { count: 0 },
      matrix: new THREE.Matrix4()
    };
  }

  parent.lastSimState.bodies.clear?.();
  parent.lastSimState.lights.clear?.();

  return [model, data, simulation, bodies, lights];
}

function configureJointMappings(demo, jointNames) {
  const model = demo.model;
  const mujoco = demo.mujoco;

  demo.policyJointNames = jointNames.slice();

  const jointTransmission = mujoco.mjtTrn.mjTRN_JOINT.value;
  const actuator2joint = [];
  for (let i = 0; i < model.nu; i++) {
    if (model.actuator_trntype[i] !== jointTransmission) {
      throw new Error(`Actuator ${i} transmission type is not mjTRN_JOINT`);
    }
    actuator2joint.push(model.actuator_trnid[2 * i]);
  }

  demo.ctrl_adr_policy = [];
  demo.qpos_adr_policy = [];
  demo.qvel_adr_policy = [];

  for (const name of jointNames) {
    const jointIdx = demo.jointNamesMJC.indexOf(name);
    if (jointIdx < 0) {
      throw new Error(`Joint "${name}" not found in MuJoCo model`);
    }
    const actuatorIdx = actuator2joint.findIndex((jointId) => jointId === jointIdx);
    if (actuatorIdx < 0) {
      throw new Error(`No actuator mapped to joint "${name}"`);
    }
    demo.ctrl_adr_policy.push(actuatorIdx);
    demo.qpos_adr_policy.push(model.jnt_qposadr[jointIdx]);
    demo.qvel_adr_policy.push(model.jnt_dofadr[jointIdx]);
  }

  demo.numActions = jointNames.length;

  demo.defaultJposPolicy = new Float32Array(demo.numActions);
  demo.defaultJposPolicy.fill(0.0);
}

export async function downloadExampleScenesFolder(mujoco) {
  const response = await fetch('./examples/scenes/files.json');
  const allFiles = await response.json();

  const requests = allFiles.map((url) => fetch('./examples/scenes/' + url));
  const responses = await Promise.all(requests);

  for (let i = 0; i < responses.length; i++) {
    let split = allFiles[i].split('/');
    let working = '/working/';
    for (let f = 0; f < split.length - 1; f++) {
      working += split[f];
      if (!mujoco.FS.analyzePath(working).exists) {
        mujoco.FS.mkdir(working);
      }
      working += '/';
    }

    if (allFiles[i].match(/\.(png|stl|skn)$/i)) {
      mujoco.FS.writeFile('/working/' + allFiles[i], new Uint8Array(await responses[i].arrayBuffer()));
    } else {
      mujoco.FS.writeFile('/working/' + allFiles[i], await responses[i].text());
    }
  }
}

export function getPosition(buffer, index, target, swizzle = true) {
  if (swizzle) {
    return target.set(
      buffer[(index * 3) + 0],
      buffer[(index * 3) + 2],
      -buffer[(index * 3) + 1]
    );
  }
  return target.set(
    buffer[(index * 3) + 0],
    buffer[(index * 3) + 1],
    buffer[(index * 3) + 2]
  );
}

export function getQuaternion(buffer, index, target, swizzle = true) {
  if (swizzle) {
    return target.set(
      -buffer[(index * 4) + 1],
      -buffer[(index * 4) + 3],
      buffer[(index * 4) + 2],
      -buffer[(index * 4) + 0]
    );
  }
  return target.set(
    buffer[(index * 4) + 0],
    buffer[(index * 4) + 1],
    buffer[(index * 4) + 2],
    buffer[(index * 4) + 3]
  );
}

export function toMujocoPos(target) {
  return target.set(target.x, -target.z, target.y);
}


function createSimulationWrapper(mujoco, model, data) {
  const force = new Float64Array(3);
  const torque = new Float64Array(3);
  const point = new Float64Array(3);

  return {
    get qpos() { return data.qpos; },
    get qvel() { return data.qvel; },
    get ctrl() { return data.ctrl; },
    get qfrc_applied() { return data.qfrc_applied; },
    get xpos() { return data.xpos; },
    get xquat() { return data.xquat; },
    get light_xpos() { return data.light_xpos; },
    get light_xdir() { return data.light_xdir; },
    get ten_wrapadr() { return model.ten_wrapadr; },
    get ten_wrapnum() { return model.ten_wrapnum; },
    get wrap_xpos() { return data.wrap_xpos; },
    step() {
      mujoco.mj_step(model, data);
    },
    resetData() {
      mujoco.mj_resetData(model, data);
    },
    forward() {
      mujoco.mj_forward(model, data);
    },
    applyForce(fx, fy, fz, tx, ty, tz, px, py, pz, bodyId) {
      force[0] = fx; force[1] = fy; force[2] = fz;
      torque[0] = tx; torque[1] = ty; torque[2] = tz;
      point[0] = px; point[1] = py; point[2] = pz;
      mujoco.mj_applyFT(model, data, force, torque, point, bodyId, data.qfrc_applied);
    },
    free() {
      data.delete();
      model.delete();
    }
  };
}
