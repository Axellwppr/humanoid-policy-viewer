import * as THREE from 'three';
import {
  quatMultiply,
  quatInverse,
  yawComponent,
  linspaceRows,
  slerpMany
} from './utils/math.js';

function clampIndex(idx, length) {
  if (idx < 0) {
    return 0;
  }
  if (idx >= length) {
    return length - 1;
  }
  return idx;
}

function toFloat32Rows(rows) {
  if (!Array.isArray(rows)) {
    return null;
  }
  return rows.map((row) => Float32Array.from(row));
}

function normalizeMotionClip(clip) {
  if (!clip || typeof clip !== 'object') {
    return null;
  }
  const jointPosRaw = toFloat32Rows(clip.joint_pos ?? clip.jointPos);
  const rootPos = toFloat32Rows(clip.root_pos ?? clip.rootPos);
  const rootQuat = toFloat32Rows(clip.root_quat ?? clip.rootQuat);
  if (!jointPosRaw || !rootPos || !rootQuat) {
    return null;
  }
  return { jointPos: jointPosRaw, rootPos, rootQuat };
}

export class TrackingHelper {
  constructor(config) {
    this.transitionSteps = config.transition_steps ?? 100;
    this.datasetJointNames = config.dataset_joint_names ?? [];
    this.policyJointNames = config.policy_joint_names ?? [];
    this.motions = {};
    this.nJoints = this.datasetJointNames.length || this.policyJointNames.length;
    this.transitionLen = 0;
    this.motionLen = 0;

    this.mapPolicyToDataset = this._buildPolicyToDatasetMap();

    for (const [name, clip] of Object.entries(config.motions ?? {})) {
      const normalized = normalizeMotionClip(clip);
      if (!normalized) {
        console.warn('TrackingHelper: invalid motion clip', name);
        continue;
      }
      this.motions[name] = normalized;
    }

    if (!this.motions.default) {
      throw new Error('TrackingHelper requires a "default" motion');
    }

    this.refJointPos = [];
    this.refRootQuat = [];
    this.refRootPos = [];
    this.refIdx = 0;
    this.refLen = 0;
    this.currentName = 'default';
    this.currentDone = true;
  }

  availableMotions() {
    return Object.keys(this.motions);
  }

  addMotions(motions, options = {}) {
    const added = [];
    const skipped = [];
    const invalid = [];
    const allowOverwrite = !!options.overwrite;

    if (!motions || typeof motions !== 'object') {
      return { added, skipped, invalid };
    }

    for (const [name, clip] of Object.entries(motions)) {
      if (!name) {
        invalid.push(name);
        continue;
      }
      if (!allowOverwrite && this.motions[name]) {
        skipped.push(name);
        continue;
      }
      const normalized = normalizeMotionClip(clip);
      if (!normalized) {
        invalid.push(name);
        continue;
      }
      this.motions[name] = normalized;
      added.push(name);
    }

    return { added, skipped, invalid };
  }

  reset(state) {
    this.currentDone = true;
    this.refIdx = 0;
    this.refLen = 0;
    this.transitionLen = 0;
    this.motionLen = 0;
    this.refJointPos = [];
    this.refRootQuat = [];
    this.refRootPos = [];
    this.currentName = 'default';
    this.requestMotion('default', state);
  }

  requestMotion(name, state) {
    if (!this.motions[name]) {
      return false;
    }
    if ((this.currentName === 'default' && this.currentDone) || name === 'default') {
      this._startMotionFromCurrent(name, state);
      return true;
    }
    return false;
  }

  isReady() {
    return this.refLen > 0;
  }

  playbackState() {
    const clampedIdx = Math.max(0, Math.min(this.refIdx, Math.max(this.refLen - 1, 0)));
    const transitionLen = this.transitionLen ?? 0;
    const motionLen = this.motionLen ?? 0;
    const inTransition = transitionLen > 0 && clampedIdx < transitionLen;
    return {
      available: this.refLen > 0,
      currentName: this.currentName,
      currentDone: this.currentDone,
      refIdx: clampedIdx,
      refLen: this.refLen,
      transitionLen,
      motionLen,
      inTransition,
      isDefault: this.currentName === 'default'
    };
  }

  advance() {
    if (this.refLen === 0) {
      return;
    }
    if (this.refIdx < this.refLen - 1) {
      this.refIdx += 1;
      if (this.refIdx === this.refLen - 1) {
        this.currentDone = true;
      }
    }
  }

  getFrame(index) {
    const clamped = clampIndex(index, this.refLen);
    return {
      jointPos: this.refJointPos[clamped],
      rootQuat: this.refRootQuat[clamped],
      rootPos: this.refRootPos[clamped]
    };
  }

  _readCurrentState(state) {
    if (state) {
      return {
        jointPos: Array.from(state.jointPos),
        rootPos: Array.from(state.rootPos),
        rootQuat: Array.from(state.rootQuat)
      };
    }

    const defaultMotion = this.motions['default'];
    const fallbackPos = defaultMotion?.rootPos?.[0] ?? new Float32Array([0.0, 0.0, 0.78]);
    const fallbackQuat = defaultMotion?.rootQuat?.[0] ?? [1.0, 0.0, 0.0, 0.0];
    const fallbackJoint = defaultMotion?.jointPos?.[0] ?? new Float32Array(this.nJoints);

    return {
      jointPos: Array.from(fallbackJoint),
      rootPos: Array.from(fallbackPos),
      rootQuat: Array.from(fallbackQuat)
    };
  }

  _alignMotionToCurrent(motion, curr) {
    const p0 = new THREE.Vector3(...motion.rootPos[0]);
    const pc = new THREE.Vector3(...curr.rootPos);

    const q0 = yawComponent(motion.rootQuat[0]);
    const qc = yawComponent(curr.rootQuat);
    const qDeltaWxyz = quatMultiply(qc, quatInverse(q0));
    const qDelta = new THREE.Quaternion(qDeltaWxyz[1], qDeltaWxyz[2], qDeltaWxyz[3], qDeltaWxyz[0]);

    const jointPos = motion.jointPos.map((row) => Float32Array.from(row));

    const offset = new THREE.Vector3(pc.x, pc.y, p0.z);
    const rootPos = motion.rootPos.map((row) => {
      const pos = new THREE.Vector3(...row);
      pos.sub(p0).applyQuaternion(qDelta).add(offset);
      return Float32Array.from([pos.x, pos.y, pos.z]);
    });

    const rootQuat = motion.rootQuat.map((row) => {
      const q = new THREE.Quaternion(row[1], row[2], row[3], row[0]);
      const aligned = qDelta.clone().multiply(q);
      return Float32Array.from([aligned.w, aligned.x, aligned.y, aligned.z]);
    });

    return { jointPos, rootQuat, rootPos };
  }

  _buildTransition(curr, firstFrame) {
    const steps = Math.max(0, Math.floor(this.transitionSteps));
    if (steps === 0) {
      return {
        jointPos: [],
        rootQuat: [],
        rootPos: []
      };
    }

    const jointPos = linspaceRows(curr.jointPos, firstFrame.jointPos[0], steps);
    const rootPos = linspaceRows(curr.rootPos, firstFrame.rootPos[0], steps);
    const rootQuat = slerpMany(curr.rootQuat, firstFrame.rootQuat[0], steps);

    return { jointPos, rootPos, rootQuat };
  }

  _startMotionFromCurrent(name, state) {
    const curr = this._readCurrentState(state);
    if (state && this.mapPolicyToDataset) {
      curr.jointPos = this._mapPolicyJointPosToDataset(curr.jointPos);
    }
    const motion = this.motions[name];
    const aligned = this._alignMotionToCurrent(motion, curr);
    const firstFrame = {
      jointPos: aligned.jointPos,
      rootQuat: aligned.rootQuat,
      rootPos: aligned.rootPos
    };

    const transition = this._buildTransition(curr, firstFrame);

    this.refJointPos = [...transition.jointPos, ...aligned.jointPos];
    this.refRootQuat = [...transition.rootQuat, ...aligned.rootQuat];
    this.refRootPos = [...transition.rootPos, ...aligned.rootPos];

    this.transitionLen = transition.jointPos.length;
    this.motionLen = aligned.jointPos.length;
    this.refIdx = 0;
    this.refLen = this.refJointPos.length;
    this.currentName = name;
    this.currentDone = this.refLen <= 1;
  }

  _buildPolicyToDatasetMap() {
    if (!this.datasetJointNames.length || !this.policyJointNames.length) {
      return null;
    }
    const datasetIndex = new Map();
    for (let i = 0; i < this.datasetJointNames.length; i++) {
      datasetIndex.set(this.datasetJointNames[i], i);
    }
    return this.policyJointNames.map((name) => {
      return datasetIndex.has(name) ? datasetIndex.get(name) : -1;
    });
  }

  _mapPolicyJointPosToDataset(jointPos) {
    if (!this.mapPolicyToDataset || !this.datasetJointNames.length) {
      return Float32Array.from(jointPos);
    }
    const out = new Float32Array(this.datasetJointNames.length);
    for (let i = 0; i < this.mapPolicyToDataset.length; i++) {
      const datasetIdx = this.mapPolicyToDataset[i];
      if (datasetIdx >= 0) {
        out[datasetIdx] = jointPos[i] ?? 0.0;
      }
    }
    return out;
  }
}
