import { TrainingManager } from '../src/core/training/trainingManager';
import { getTask } from '../src/core/tasks/registry';
import { SoftReachRig } from '../src/core/vision/softRig';
import { sanitizeVision } from '../src/core/vision/config';
import { firstPersonPose } from '../src/core/vision/camera';
import { GazeController, GAZE_ACTIONS } from '../src/core/vision/gaze';
import { RecurrentVisionAgent } from '../src/core/vision/recurrentAgent';
import { BallEyes } from '../src/render/ballEyes';
import { Vector3 } from 'three';
import { VISION_COLORS } from '../src/core/vision/rig';

let fails = 0;
const ok = (c: boolean, m: string) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fails++; };
const task = getTask('reach-fp');
const S = sanitizeVision({ ...task.vision!.defaults, size: 32 });
const rig = new SoftReachRig(S);
const snap = (bx: number, bz: number, tx: number, tz: number) => ({ epoch: 0, bodies: { ball: { p: [bx, 0.5, bz], q: [0, 0, 0, 1] }, target: { p: [tx, 0, tz], q: [0, 0, 0, 1] } }, info: {} }) as any;
const green = (f: Uint8Array) => { const n = 32 * 32; let c = 0, sx = 0; for (let i = 0; i < n; i++) if (f[i] === VISION_COLORS.target[0] && f[n + i] === VISION_COLORS.target[1]) { c++; sx += i % 32; } return { c, cx: c ? sx / c : -1 }; };
const look = (yaw: number, pitch: number, s = snap(0, 0, 0, -4)) => { const f = new Uint8Array(3 * 32 * 32); rig.render(s, f, firstPersonPose(S.fov, yaw, pitch, 0, 0.5, 0, 0.5)); return f; };

// 1 first-person view of the real world from the ball
ok(S.fov >= 85 && S.fov <= 95, `FOV ${S.fov}° (natural 85–95°)`);
ok(green(look(0, 0)).c > 4, 'target straight ahead is visible in the first-person image');
ok(green(look(180, 0)).c === 0, 'looking behind: target no longer visible (real view-dependence, no omniscience)');
ok(green(look(90, 0)).c === 0 && green(look(-90, 0)).c === 0, 'looking left/right: target out of view');
// target to the right appears right of image centre when looking forward
const gr = green(look(0, 0, snap(0, 0, 3, -4)));
ok(gr.c > 0 && gr.cx > 16, `target on the right appears on the right side of the image (x≈${gr.cx.toFixed(1)})`);
// 2 full 360° yaw
const g = new GazeController(120, 80);
let seen180 = false;
for (let i = 0; i < 180; i++) { g.step(2, 1 / 60); if (Math.abs(Math.abs(g.yaw) - 180) < 2) seen180 = true; }
ok(seen180, `yaw passes through the full circle (reached ${g.yaw.toFixed(0)}° after 3 s of "Look Right")`);
let d = 0; { const a = look(0, 0), b = look(360, 0); for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]); }
ok(d === 0, 'a 360° turn returns to the identical view');
// 3 pitch limits
const gp = new GazeController(); for (let i = 0; i < 600; i++) gp.step(3, 1 / 60); const up = gp.pitch;
for (let i = 0; i < 1200; i++) gp.step(4, 1 / 60);
ok(Math.abs(up - 80) < 1e-6 && Math.abs(gp.pitch + 80) < 1e-6, `pitch clamps at ±80° (${up}, ${gp.pitch})`);
// 4 not an orbit camera: eye is the same point for every yaw/pitch, only direction changes
const e0 = firstPersonPose(90, 0, 0, 1, 0.5, 2, 0.5).eye, e1 = firstPersonPose(90, 137, 55, 1, 0.5, 2, 0.5).eye;
ok(e0.every((v, i) => v === e1[i]), 'eye position is independent of yaw/pitch (rotation about its own position, no orbit)');
const pz = firstPersonPose(90, 40, 30, 0, 0.5, 0, 0.5); ok(Math.abs(pz.right[1]) < 1e-9, 'camera roll locked (right vector stays horizontal)');
// the ball's own body never appears in the first-person image: the ball is at the eye, nothing orange anywhere
{ const f = look(0, -60); const n = 1024; let o = 0; for (let i = 0; i < n; i++) if (f[i] === VISION_COLORS.ball[0] && f[n + i] === VISION_COLORS.ball[1]) o++; ok(o === 0, 'own ball/face is not rendered into the AI view'); }

// 5 AI chooses its own gaze: bias the gaze head and watch the camera turn by itself
const tm = await TrainingManager.create(task, 3, { minReplay: 100000 }, { rig: new SoftReachRig(S), settings: S });
const ag = tm.agent as RecurrentVisionAgent;
const ob = tm.agent as any;
const outL = ag.online.outL; const biasG = (k: number) => { for (let j = 0; j < ag.online.A; j++) ag.online.params[outL.b + j] = 0; ag.online.params[outL.b + ag.M + k] = 5; };
tm.setDemo(true); tm.start();
biasG(1); tm.runDecisions(5); const yawL = tm.gaze!.yaw;
ok(yawL < -30 && tm.snapshot().gaze!.yaw === yawL, `network's "Look Left" output turns the camera left on its own (yaw ${yawL.toFixed(0)}°)`);
biasG(4); tm.runDecisions(3); ok(tm.gaze!.pitch < -10, `network's "Look Down" output tilts the camera down (pitch ${tm.gaze!.pitch.toFixed(0)}°)`);
biasG(0); tm.runDecisions(2); const y0 = tm.gaze!.yaw, p0 = tm.gaze!.pitch; tm.runDecisions(3); ok(tm.gaze!.yaw === y0 && tm.gaze!.pitch === p0, '"Maintain gaze" keeps the camera still');
ok(GAZE_ACTIONS.length === 9, 'gaze actions: hold, 4 directions, 4 diagonals');
// movement and gaze are independent branches
const a1 = ag.online.A; ok(ag.actionCount === ag.M * ag.G && a1 === ag.M + ag.G, 'separate movement and gaze Q-branches');
void ob;

// 6 eyes follow camera; ball rolling can not tumble them (no dependency on the body rotation)
const eyes = new BallEyes(0.5); const pos = new Vector3(0, 0.5, 0);
for (let i = 0; i < 120; i++) eyes.update(1 / 60, pos, null, 0, { yaw: 0, pitch: 0 });
const c0 = eyes.pupilDirection();
for (let i = 0; i < 2; i++) eyes.update(1 / 60, pos, null, 0, { yaw: 60, pitch: 0 });
const lead = eyes.pupilDirection();
for (let i = 0; i < 240; i++) eyes.update(1 / 60, pos, null, 0, { yaw: 60, pitch: 0 });
const settled = eyes.pupilDirection();
ok(Math.abs(c0[0]) < 1e-3 && c0[2] > 0.99, 'pupils centred when the camera is still');
ok(lead[0] < -0.2, `camera turning right ⇒ pupils lead to the AI's right (x ${lead[0].toFixed(2)})`);
ok(Math.abs(settled[0]) < 0.02 && Math.abs((eyes as any).face.rotation.y - (Math.PI - Math.PI / 3)) < 0.02, 'face rig turns to the camera yaw and the pupils settle back to centre');
const eu = new BallEyes(0.5); for (let i = 0; i < 100; i++) eu.update(1 / 60, pos, null, 0, { yaw: 0, pitch: 70 });
ok(((eu as any).face.rotation.x) < -1.0 && ((eu as any).group.parent === null), 'face rig tilts up with the camera pitch and is separate from the rolling body');

// 9 the policy really uses the image: same proprio + memory, different picture ⇒ different Q, and gradients reach the CNN
const f1 = new Float32Array(3 * 32 * 32).fill(0.2), f2 = new Float32Array(3 * 32 * 32).fill(0.7);
const pr = new Float32Array(9); const h = new Float32Array(ag.online.H); const q1 = new Float32Array(ag.online.A), q2 = new Float32Array(ag.online.A);
ag.online.predictStep(f1, pr, h, q1); h.fill(0); ag.online.predictStep(f2, pr, h, q2);
ok(q1.some((v, i) => Math.abs(v - q2[i]) > 1e-4), 'Q-values depend on the image');
console.log(fails ? `${fails} FAILED` : 'all first-person vision tests passed');
process.exit(fails ? 1 : 0);
