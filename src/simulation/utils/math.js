export function normalizeQuat(quat) {
  const [w, x, y, z] = quat;
  const n = Math.hypot(w, x, y, z);
  if (n < 1e-9) {
    return [1, 0, 0, 0];
  }
  const inv = 1.0 / n;
  return [w * inv, x * inv, y * inv, z * inv];
}

export function quatConjugate(quat) {
  return [quat[0], -quat[1], -quat[2], -quat[3]];
}

export function quatMultiply(a, b) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw
  ];
}

export function quatInverse(quat) {
  const conj = quatConjugate(quat);
  const [w, x, y, z] = quat;
  const normSq = w * w + x * x + y * y + z * z;
  if (normSq < 1e-9) {
    return [1, 0, 0, 0];
  }
  const inv = 1.0 / normSq;
  return [conj[0] * inv, conj[1] * inv, conj[2] * inv, conj[3] * inv];
}

export function yawComponent(quat) {
  const [w, x, y, z] = quat;
  const sinyCosp = 2.0 * (w * z + x * y);
  const cosyCosp = 1.0 - 2.0 * (y * y + z * z);
  const yaw = Math.atan2(sinyCosp, cosyCosp);
  const half = 0.5 * yaw;
  return normalizeQuat([Math.cos(half), 0.0, 0.0, Math.sin(half)]);
}

export function linspaceRows(a, b, steps) {
  if (steps <= 0) {
    return [];
  }
  const result = [];
  const denom = steps + 1;
  for (let i = 1; i <= steps; i++) {
    const t = i / denom;
    const row = new Float32Array(a.length);
    for (let j = 0; j < a.length; j++) {
      row[j] = (1.0 - t) * a[j] + t * b[j];
    }
    result.push(row);
  }
  return result;
}

export function slerpMany(q0, q1, steps) {
  if (steps <= 0) {
    return [];
  }
  const start = normalizeQuat(q0);
  let end = normalizeQuat(q1);
  let dot = start[0] * end[0] + start[1] * end[1] + start[2] * end[2] + start[3] * end[3];
  if (dot < 0.0) {
    dot = -dot;
    end = end.map((v) => -v);
  }

  const results = [];
  const EPS = 1e-6;
  if (1.0 - dot < EPS) {
    const denom = steps + 1;
    for (let i = 1; i <= steps; i++) {
      const t = i / denom;
      const row = new Float32Array(4);
      for (let j = 0; j < 4; j++) {
        row[j] = (1.0 - t) * start[j] + t * end[j];
      }
      results.push(Float32Array.from(normalizeQuat(row)));
    }
    return results;
  }

  const omega = Math.acos(dot);
  const sinOmega = Math.sin(omega);
  const denom = steps + 1;
  for (let i = 1; i <= steps; i++) {
    const t = i / denom;
    const coeff0 = Math.sin((1.0 - t) * omega) / sinOmega;
    const coeff1 = Math.sin(t * omega) / sinOmega;
    const row = new Float32Array(4);
    for (let j = 0; j < 4; j++) {
      row[j] = coeff0 * start[j] + coeff1 * end[j];
    }
    results.push(Float32Array.from(row));
  }
  return results;
}

export function clampFutureIndices(base, steps, length) {
  return steps.map((step) => {
    const idx = base + step;
    if (idx < 0) return 0;
    if (idx >= length) return length - 1;
    return idx;
  });
}

export function toFloatArray(value, length, fallback = 0.0) {
  if (Array.isArray(value)) {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      out[i] = value[i] ?? fallback;
    }
    return out;
  }
  if (ArrayBuffer.isView(value)) {
    const out = new Float32Array(length);
    const src = value;
    for (let i = 0; i < length; i++) {
      out[i] = src[i] ?? fallback;
    }
    return out;
  }
  const out = new Float32Array(length);
  out.fill(value ?? fallback);
  return out;
}

export function quatApplyInv(quat, vec) {
  const [w, x, y, z] = quat;
  const vx = vec[0];
  const vy = vec[1];
  const vz = vec[2];
  const tx = 2.0 * (y * vz - z * vy);
  const ty = 2.0 * (z * vx - x * vz);
  const tz = 2.0 * (x * vy - y * vx);
  const cx = y * tz - z * ty;
  const cy = z * tx - x * tz;
  const cz = x * ty - y * tx;
  return [
    vx - w * tx + cx,
    vy - w * ty + cy,
    vz - w * tz + cz
  ];
}

export function quatToRotVec(quat) {
  const q = normalizeQuat(quat);
  const w = Math.max(-1.0, Math.min(1.0, q[0]));
  const angle = 2.0 * Math.acos(w);
  const s = Math.sqrt(Math.max(0.0, 1.0 - w * w));
  if (s < 1e-6) {
    return [q[1] * 2.0, q[2] * 2.0, q[3] * 2.0];
  }
  const inv = 1.0 / s;
  return [q[1] * inv * angle, q[2] * inv * angle, q[3] * inv * angle];
}

export function quatToRot6d(quat) {
  const [w, x, y, z] = normalizeQuat(quat);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  const r00 = 1.0 - 2.0 * (yy + zz);
  const r01 = 2.0 * (xy - wz);
  const r10 = 2.0 * (xy + wz);
  const r11 = 1.0 - 2.0 * (xx + zz);
  const r20 = 2.0 * (xz - wy);
  const r21 = 2.0 * (yz + wx);

  return [r00, r10, r20, r01, r11, r21];
}
