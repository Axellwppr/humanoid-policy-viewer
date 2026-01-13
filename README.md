# Humanoid Policy Viewer

Single-page Vue 3 + Vuetify app that runs a MuJoCo WebAssembly scene in the
browser and drives it with an ONNX policy. The default setup loads the G1
scene, policy, and motion clips from `public/examples`.

Demos: [Humanoid Policy Viewer](https://motion-tracking.axell.top/), [GentleHumanoid Web Demo](https://gentle-humanoid.axell.top/)

## Quick start

```bash
npm install
npm run dev
```

## Project structure

- `src/views/Demo.vue` - UI controls for the live demo
- `src/simulation/main.js` - bootstraps MuJoCo, Three.js renderer, and policy loop
- `src/simulation/mujocoUtils.js` - scene/policy loading utilities and filesystem preloading
- `src/simulation/policyRunner.js` - ONNX inference wrapper and observation pipeline
- `node_modules/mujoco-js/` - MuJoCo wasm runtime (npm package)
- `public/examples/scenes/` - MJCF files + meshes staged into MuJoCo's MEMFS
- `public/examples/checkpoints/` - policy config JSON, ONNX file, and motion clips

## Add your own robot, policy and motions

1. Add your MJCF + assets.
   - Create `public/examples/scenes/<robot>/`.
   - Put your MJCF as `public/examples/scenes/<robot>/<robot>.xml`.
   - Add all meshes/textures used by the MJCF into the same folder.
   - Append every file path to `public/examples/scenes/files.json` so the
     loader can preload them into `/working/` in the wasm filesystem.

2. Add your policy config and ONNX.
   - Create `public/examples/checkpoints/<robot>/tracking_policy.json`.
   - Place the ONNX model at `public/examples/checkpoints/<robot>/tracking_policy.onnx`.
   - In the JSON, make sure these fields are correct:
     - `onnx.path` points to your ONNX file (example: `./examples/checkpoints/<robot>/tracking_policy.onnx`)
     - `policy_joint_names` matches the joint names in your MJCF actuators
     - `obs_config` uses observation names that exist in `src/simulation/observationHelpers.js`
     - `action_scale`, `stiffness`, `damping`, and `default_joint_pos` lengths
       match `policy_joint_names`
   - You need to adapt the observation helper functions in
     `src/simulation/observationHelpers.js` if your policy uses
     different observations than the built-in ones, and modify `src/simulation/policyRunner.js` to control the robot.

3. (Optional) Add tracking motions.
   - Add an index at `public/examples/checkpoints/<robot>/motions.json`.
   - Put per-motion clips in `public/examples/checkpoints/<robot>/motions/`.
   - In `tracking_policy.json`, set `tracking.motions_path` to the index file.
   - The app downloads all motion clips listed in the index when the policy loads.
   - The index uses this shape:
     - `format`: `tracking-motion-index-v1`
     - `base_path`: relative path to the motions folder (example: `./motions`)
     - `motions`: list of `{ name, file }` entries
   - Each motion clip file must include a `default` clip overall and each clip contains:
     - `joint_pos` (or `jointPos`): per-frame joint arrays
     - `root_pos` (or `rootPos`): per-frame root positions
     - `root_quat` (or `rootQuat`): per-frame root quaternions (w, x, y, z)

4. Point the app to your robot and policy.
   - Update `src/simulation/main.js`:
     - `this.currentPolicyPath = './examples/checkpoints/<robot>/tracking_policy.json'`
     - `await this.reloadScene('<robot>/<robot>.xml')`
     - `await this.reloadPolicy('./examples/checkpoints/<robot>/tracking_policy.json')`

If you want to keep multiple robots around, you can expose a selector in
`src/views/Demo.vue` and call `demo.reloadScene(...)` and `demo.reloadPolicy(...)`
from there.
