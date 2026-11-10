# BALLMIND — Physics-Based Reinforcement Learning Simulator

**▶ เล่น / Play online: https://yahahahau.github.io/BALLMIND/**

ลูกบอล 3 มิติที่มี "สมอง" เป็น Neural Network เรียนรู้ภารกิจด้วยการลองผิดลองถูกในฟิสิกส์จริง — ทั้งหมดทำงานในเบราว์เซอร์ ไม่ต้องมีเซิร์ฟเวอร์ Python หรือบริการ AI ภายนอก ออกแบบให้ใช้งานบน iPad (Safari) เป็นหลัก

## ภารกิจ: REACH THE SQUARE
AI ต้องควบคุมลูกบอลให้ไปสัมผัสสี่เหลี่ยมเป้าหมายโดยไม่ตกขอบสนาม เป้าหมายสุ่มตำแหน่งใหม่ทุก Episode เมื่อสำเร็จหรือตกสนามระบบจะ Reset อัตโนมัติ โดยไม่ล้างสมองหรือประสบการณ์ที่สะสม

## วิธีใช้
1. เปิดลิงก์ด้านบน เลือกภารกิจในหน้าต่าง **SELECT TRAINING TASK**
2. กด **Start Training** แล้วเลือกความเร็ว (1× – MAX) ดู AI พัฒนาขึ้นจากกราฟ Reward / Success Rate
3. **Demo** = ให้ AI ใช้นโยบายที่ดีที่สุด (ε = 0) โดยไม่เรียนรู้เพิ่ม
4. ลากนิ้วหมุนกล้อง · จีบนิ้วซูม · ปุ่ม Camera รีเซ็ตมุมมอง
5. ความคืบหน้าบันทึกอัตโนมัติใน IndexedDB ของเบราว์เซอร์ · **Export / Import Model** เป็นไฟล์ JSON · **Reset AI Training** ล้างทั้งหมดจริง

## ผลทดสอบการเรียนรู้ (headless, `npm run test:learn`)
| | สำเร็จ (greedy) |
|---|---|
| ก่อนเทรน | 13–26% |
| หลังเทรน 20,000 สเต็ป (3 seeds) | **100%** |

เวลาเฉลี่ยในการเข้าเป้าลดจาก ~5 วินาที เหลือ ~1.3 วินาที

## ภารกิจ 02: OBBY CHALLENGE
สนามอุปสรรค 5 ด่าน (THE GAP → NARROW PATH → MOVING PLATFORMS → ROTATION & PRECISION → THE FINAL GAUNTLET) แต่ละด่านมีจุดเริ่ม เช็คพอยต์ และเส้นชัย · แพลตฟอร์มเคลื่อนที่/หมุนและแท่งกวาดเป็น kinematic body ของ Rapier ที่ชนกับลูกบอลจริง (ไม่มี teleport ข้ามสิ่งกีดขวาง; ตกแล้วเกิดใหม่ที่เช็คพอยต์ล่าสุดโดยสมองและ replay buffer ยังอยู่)

- **สมองเดียวทุกด่าน**: DQN เดิม (Double DQN, target network, replay, ε-greedy) · Observation 83 ค่า (สถานะลูกบอล, แพลตฟอร์มถัดไป, เซ็นเซอร์ raycast พื้น/ข้าง/ใต้เท้า, สิ่งกีดขวางที่เคลื่อนที่, เช็คพอยต์, ความคืบหน้า) · 15 Action · MLP 83→96→96→15
- **Reward**: potential-based shaping ตามความคืบหน้าบนเส้นทาง + รางวัลลงจอด/เช็คพอยต์ (ครั้งเดียวต่อรอบ) + เส้นชัย − ตก − เวลา · ตกแล้วศักย์ลดลง จึงไม่มีช่องโกงวนลูป
- **Curriculum**: ปลดล็อกด่านถัดไปเมื่ออัตราสำเร็จ/ความคืบหน้า/เวลาเฉลี่ยผ่านเกณฑ์ (ปรับได้ในหน้า) · ซ้อมด่านเก่า 25% กันลืม · เลือกด่านเองได้ · เริ่มจากเช็คพอยต์ได้ (นับเป็นการฝึก ไม่นับเป็นสำเร็จทั้งด่าน)
- **Dashboard**: 15 ค่าสถิติ + กราฟจริง 5 เส้น (Reward, Success Rate, Completion Time, Checkpoint Progress, Training Loss) แยกตามด่าน · กล้องตามบอล + Overview Camera
- **Training Replay**: เลือก episode, เล่น/หยุด/ซ้ำ/Slow-motion/ไทม์ไลน์, จุดเช็คพอยต์และจุดตกบนเส้นทาง, เปรียบเทียบ episode เก่ากับใหม่ (บันทึกเฉพาะตำแหน่งบอล เพราะแพลตฟอร์มเป็นฟังก์ชันของเวลา)
- **บันทึก/โหลด**: IndexedDB เก็บ weights, target, replay buffer (int8), สถิติ, curriculum, loss, replays · Export/Import JSON มีตรวจเวอร์ชัน/ขนาดโมเดลก่อนโหลด

### ผลเทรนจริง (headless `npx tsx scripts/train-obby.ts`, ไม่มีสคริปต์ช่วย)
ด่าน 1 ผ่านได้ที่ ~70k สเต็ป, ด่าน 2 เริ่มผ่านที่ ~600k, ด่าน 3 ปลดล็อกและเริ่มผ่านที่ ~1M สเต็ป (ดูตัวเลขล่าสุดในรายงานของงานนี้) ด่าน 4–5 ยังต้องเทรนต่อ — ระบบแยกชัดเจนว่า "AI ยังไม่เก่ง" ต่างจาก "ระบบพัง" ด้วย `scripts/test-obby-physics.ts` และ `scripts/test-training.ts`

## 3D World · Skybox & Atmosphere
- ค่าเริ่มต้นเป็นท้องฟ้าสีฟ้าธรรมดา (CC0 จากแพ็ก Screaming Brain Studios — ดู `src/assets/SKY-LICENSE.txt`)
- **Import Skybox** (ในแผง *3D WORLD · SKYBOX*): `.hdr` (HDRLoader) · `.exr` (EXRLoader) · แพนอรามา 2:1 `.png/.jpg` · cubemap 6 ภาพ (+X −X +Y −Y +Z −Z เลือกพร้อมกัน) — ตรวจเนื้อไฟล์จริง ไม่ใช้ชื่อไฟล์ และตรวจขนาด/อัตราส่วน/เหลี่ยมจัตุรัส/ขนาดเท่ากัน มี Preview ก่อน Apply, Remove, Restore Default และจำท้องฟ้าที่เลือกไว้ (IndexedDB, ไฟล์ไม่ออกเน็ต)
- Background + PMREM environment (`scene.environment`) แยกกันปรับได้ · แสงหลัก/สีแสงรอบทิศ/ทิศแสงอาทิตย์/สีหมอกคำนวณจากภาพท้องฟ้าที่นำเข้า (ปิดได้ด้วย *Match lighting to sky*)
- สไลเดอร์: Skybox Brightness, Environment Lighting, Exposure, Contrast, Saturation, Ambient, Shadow Intensity — กระทบแค่การเรนเดอร์ ไม่แตะฟิสิกส์/AI

## เพลงประกอบ
แผง **MUSIC** เล่น 3 เพลง (Veil44 – Pulse, diedlonely – bipolar, Victoriya – Two Years; ไฟล์เดิมจากผู้ใช้ บีบเป็น 112 kbps ใน `public/music/`) · โหมด **Loop** เล่นวน 1→2→3→1 หรือ **Shuffle** สุ่มภายในสามเพลงนี้ (ไม่ซ้ำเพลงเดิมติดกัน) · เลือกเพลง/ข้าม/เลื่อนเวลา/ปรับเสียง และจำค่าไว้ · เบราว์เซอร์บล็อกเสียงอัตโนมัติ จึงเริ่มเล่นหลังกดปุ่ม Play (ถ้าครั้งก่อนเล่นอยู่ จะต่อให้เมื่อแตะหน้าจอครั้งแรก)

## Graphics · GTAO (Ambient Occlusion)
GTAO จริงด้วย `GTAOPass` ของ Three.js: RenderPass (MSAA, half-float) → GTAO + Poisson denoise → OutputPass · ปิดแล้ว dispose composer ทั้งหมดและกลับไปใช้ `renderer.render` เดิม · Shadow map ไม่ถูกแตะ
- ตั้งค่าในแผง MODEL & DISPLAY: ON/OFF, Quality (Low 8 / Medium 12 / High 20 / Ultra 32 samples + ขนาด buffer + denoiser ต่างกันจริง), Intensity, Radius, Resolution scale, Performance Mode (ครึ่งจำนวน sample, เพดาน ½-res, ไม่มี MSAA), Adaptive resolution (ลดความละเอียด AO อัตโนมัติเมื่อเฟรมช้า), Reset to Default
- ค่าเริ่มต้น: ON · Medium · Intensity 0.8 · Performance OFF · บันทึกในระบบ settings เดิม · ถ้า GTAO error จะปิดตัวเองและแจ้งข้อความ แล้วเรนเดอร์ต่อแบบปกติ

## AI Vision · ภารกิจ "REACH THE SQUARE · VISION"

ลูกบอลมองโลกผ่าน **กล้องของ AI เอง** (แยกจากกล้องผู้เล่นโดยสิ้นเชิง) แล้วเรียนรู้ด้วย CNN ที่เทรนจากผลลัพธ์ฟิสิกส์จริงของ Rapier:

```
AI Camera → RGB Image (stack 2 เฟรม) → CNN Visual Encoder → Visual Features ┐
Physics sensors (ความเร็ว, ความเร็วเชิงมุม, แตะพื้น) ─────────────────────────┴→ Dense → Q-values → Action → Rapier
```

- **กล้อง AI**: เรนเดอร์ geometry จริงของฉาก (พื้น, เป้า, ลูกบอล) ด้วย WebGL2 บน `OffscreenCanvas` *ใน worker* ลงเป้า 40×40 RGB (ปรับ 16–64) มุมกล้องเกาะตามตำแหน่งบอล (ไม่หมุนตามบอล) ปรับ FOV / pitch / ทิศ / ระยะ ได้ในแผง **AI VISION**; ภาพที่ AI เห็น **ซ่อนเป็นค่าเริ่มต้น** (กดเปิดเพื่อดูภาพจริงที่ส่งเข้าเน็ตเวิร์ก ~5 Hz และจะไม่ถูกส่งจาก worker ถ้าไม่เปิด)
- **ไม่มีพิกัด**: observation ของโหมดนี้มีแค่ 8 ค่าจากเซ็นเซอร์ของตัวบอล (ความเร็ว 3, ความเร็วเชิงมุม 3, แตะพื้น, speed) – ไม่มีตำแหน่งเป้า ตำแหน่งบอล หรือระยะทางให้ AI ต้อง "เห็น" เป้าจากภาพ (ส่วน *reward* ยังคำนวณจากฟิสิกส์จริงเหมือนโหมดปกติ)
- **เฉพาะ perception**: ภาพเป็นสีเรียบ unlit – ไม่มีเงา, GTAO, skybox หรือแสง ดังนั้นการตั้งค่ากราฟิกของผู้เล่นไม่กระทบสิ่งที่ AI เห็น
- **CNN เขียนเอง** (`src/core/vision/visionNet.ts`): conv4×4/2(8) → conv3×3/2(16) → conv3×3/1(16) → dense 64 (visual features) → ต่อกับ physics sensors → dense 64 → Q ; backprop เขียนเอง ผ่าน gradient check (`npx tsx scripts/test-visionnet.ts`) ~60k พารามิเตอร์ ~0.54M MAC/forward
- **Temporal-ready**: stack เฟรมก่อนหน้า (ค่าเริ่มต้น 2) – เปลี่ยน dense head เป็น GRU ได้ภายหลังโดยไม่แตะส่วนอื่น
- **Replay เก็บเฟรม uint8 ครั้งเดียว** ใน frame pool (transition เก็บเฉพาะ id) – 12,000 transitions ≈ 40 MB ; replay ของ vision ไม่ถูกบันทึกลง IndexedDB (น้ำหนักเน็ต/ความคืบหน้าบันทึกตามปกติ)
- **Fallback**: ถ้า worker ไม่มี OffscreenCanvas/WebGL2 จะใช้กล้อง CPU ที่คำนวณด้วยโมเดลกล้องเดียวกัน (ช้ากว่า, แสดงข้อความในแผง) ทรัพยากร GPU (render target, geometry, context) ถูก dispose เมื่อเปลี่ยนภารกิจ/ตั้งค่า
- ทดสอบ: `npx tsx scripts/test-vision.ts` (pipeline) · `npx tsx scripts/vision-learn.ts 32000 40` (เทรนจริง)

### ผลเทรนจริง (headless, กล้อง CPU ตัวอ้างอิง, ไม่มีสคริปต์ช่วย)

เทรนจากภาพล้วน + เซ็นเซอร์ 32,000 decision (~16 นาที บน CPU Node): success rate (50 ตอนล่าสุด) อยู่ที่ ~2–30% ช่วงสุ่ม/ε สูง (ไม่มีการเรียนรู้ใน ~20k steps แรก) แล้วพุ่งเป็น **~90%** เมื่อ ε ลดเหลือ 0.08 ที่ ~24–26k steps ; โหมด Demo (greedy ไม่สำรวจ) 184 ตอน **สำเร็จ 90%** ผลนี้มาจากเน็ตเวิร์กที่เห็นภาพเท่านั้น (ไม่มีพิกัด)
ข้อจำกัดที่ยังไม่ได้ยืนยัน: ยังไม่ได้วัดความเร็วบน iPad จริง ; ในเบราว์เซอร์ SwiftShader (GL ซอฟต์แวร์) การอ่านพิกเซลจาก worker ช้ามากเมื่อฉากหลักเรนเดอร์พร้อมกัน (ไม่ใช่ตัวแทนของ GPU จริง) ; vision สำหรับ Obby ยังไม่ได้ทำ (ภารกิจ Obby ยังใช้ observation แบบเดิม)

## AI Eyes · Gaze Tracking

ลูกบอลมีตา 3 มิติสองข้าง (ตาขาว + รูม่านตาทรงกลมแบน + ประกายตา) เป็น **ส่วนแสดงผลล้วน ๆ** – ไม่มี collider, ไม่ตกเงา, ไม่แตะฟิสิกส์/การเรียนรู้ (`src/render/ballEyes.ts`)

- รูม่านตาขยับจาก **เวกเตอร์ forward จริงของกล้อง AI** (ฟังก์ชัน `visionCameraPose` ตัวเดียวกับที่ worker ใช้เรนเดอร์ภาพให้ AI) → แปลงเป็นพิกัดท้องถิ่นของใบหน้า → ได้ทิศมอง normalize → ย้ายรูม่านตาบนผิวตาขาว (จำกัดมุมไม่ให้หลุดขอบ/ทะลุ) พร้อม smoothing ; ซ้าย/ขวา/บน/ล่าง/เฉียง/ตรง ตรงกับทิศกล้องจริง (ตรวจด้วยเทสต์ใน Chromium)
- "หน้า" ของบอลเป็น anchor แยกจาก mesh ที่หมุนตามฟิสิกส์ จึงไม่กลิ้งตามบอล; ใบหน้าหันตามทิศที่บอลเคลื่อนที่ (เริ่มต้นหัน +Z)
- ค่าเริ่มต้นของกล้อง AI คือมองลงตรง ๆ (pitch 90°) รูม่านตาจึงมองลงล่าง – ปรับ pitch/ทิศ ในแผง AI VISION แล้วตาจะเปลี่ยนตามหลังกด Apply ; ภารกิจที่ไม่มีกล้อง AI (Reach/Obby ปกติ) ตามองตรงไปข้างหน้า (ไม่มีการสุ่ม/สคริปต์)
- ทำงานทั้งตอนเทรนและ Replay (ใช้ลูกบอลเดียวกัน) ; ไม่แสดงภาพจากกล้อง AI ให้ผู้เล่น ; เรขาคณิตเบา (3 mesh ต่อข้าง) ไม่มี post-processing เพิ่ม
- ข้อจำกัด: กล้อง AI ยังเป็นแบบ world-aligned ที่ไม่ได้ถูก AI ควบคุมเอง จึงยังไม่มีการ "เหลือบมอง" โดย AI ; ตาแสดงทิศของกล้องตามค่าที่ตั้ง

## AI Vision V2 · First-Person (REACH THE SQUARE · FIRST-PERSON)

โมเดลใหม่แยกจากของเดิมทั้งหมด (task `reach-fp`, ช่องเก็บโมเดลแยก) – agent แบบ sensor เดิมและ Vision v1 (มุมมองบน) ยังเลือกใช้ได้และไม่ถูกแตะ ; เปิดเว็บครั้งแรกระบบจะ **สำรองโมเดลเดิมทุกตัวไปไว้ที่ `backup:v1:*` ใน IndexedDB** (ไม่เขียนทับ ไม่ลบ) พร้อมฟังก์ชัน `restoreLegacyBackup`

```
First-person eye camera → RGB (48×48) → CNN → visual features ┐
Physics proprioception (9 ค่าของตัวบอลเอง) ─────────────────────┴→ GRU (หน่วยความจำ) → Q-movement + Q-gaze → Rapier + กล้องหมุน
```

- **กล้องมุมมองบุคคลที่หนึ่ง**: perspective จริง FOV 90°, เห็นฉากจริง (depth/occlusion ผ่านเรนเดอร์จริง), ตัวบอล/ลูกตาไม่ถูกเรนเดอร์เข้าภาพ AI, หมุนรอบตำแหน่งตัวเอง (ไม่ orbit) yaw 360°, pitch ±80°, ล็อก roll
- **AI ควบคุมการมองเอง**: action ชุดแยก 9 แบบ (คงที่, ซ้าย, ขวา, ขึ้น, ลง, 4 เฉียง) หมุนต่อเนื่อง 120°/s ; ไม่มีการล็อกเป้า/พิกัดลับ ; การเดินเป็น "สัมพัทธ์กับทิศที่มอง" (เหมือนเกม FPS) แต่ตัดสินใจแยกจากการมอง (Q สองสาขา, branching Double DQN)
- **CNN + GRU เทรนร่วมกัน** (recurrent Double DQN, ลำดับยาว 8, burn-in 3, BPTT เขียนเอง – ผ่าน gradient check `scripts/test-recurrent.ts`)
- **Observation ไม่มีข้อมูลลับ**: ความเร็ว (ในกรอบกล้อง), ความเร็วเชิงมุม, แตะพื้น, speed, pitch ของตัวเอง ; เป้า/พิกัดใช้เฉพาะใน reward
- **Face rig / ตา**: ใบหน้าหันตามทิศกล้อง (yaw/pitch), รูม่านตาแสดงส่วนต่างระหว่างทิศกล้องจริงกับทิศใบหน้า (นำหน้าเมื่อหันแล้วกลับกลางเมื่อใบหน้าตามทัน) ; แยกจากการกลิ้งของลูกบอล
- **แก้ jitter**: worker ประทับเวลาฟิสิกส์ (fixed 1/60 s) ใน snapshot, viewport ใช้ snapshot interpolation (ลูกบอล, ตา, เป้า, แพลตฟอร์ม Obby ใช้สถานะเดียวกัน) แทนการ "ไล่" แบบ exponential ; ไม่แตะ state ของ Rapier ; ทดสอบจำลองการมาของ snapshot ที่ไม่สม่ำเสมอ: ความแกว่งต่อเฟรมลดจาก sd 0.0103 → 0.0041 ม. (`scripts/test-interp.ts`)
- **Developer debug** (ปิดเป็นค่าเริ่มต้น): พรีวิวกล้อง, yaw/pitch, gaze action, อัตรา vision, สถานะ input, lag การซิงก์ ; ปิดได้โดยไม่กระทบ AI (ภาพจะไม่ถูกส่งจาก worker ถ้าปิด)
- **ทดสอบ**: `scripts/test-vision2.ts` (21 ข้อ: มองรอบตัว 360°, จำกัด pitch, ไม่ orbit, AI เลือกมองเอง, ตา/pupil ตามกล้อง, Q ขึ้นกับภาพ ฯลฯ)

### ผลเทรนจริง (กล้อง CPU ตัวอ้างอิง, ไม่มีสคริปต์ช่วย)
เทรน ~80,000 decision (~8,200 updates, ~35 นาทีบน CPU): success rate ช่วงสุ่ม/ε สูง ~10–28% → เริ่มดีขึ้นหลัง ~40k steps → 46–68% ระหว่างเทรน (ε=0.08). ประเมินแบบ greedy ที่ checkpoint ~74k steps: **สำเร็จ 81% (83 ตอน)** ; ถ้า **บังตาให้กล้อง (ภาพเทาคงที่) สำเร็จ 0% (80 ตอน)** → นโยบายใช้ภาพจริงและใช้ภาพจริง ๆ (`scripts/vision-ablate.ts`)
ข้อจำกัด: ยังไม่ได้ทดสอบบน iPad และยังไม่ได้เทรนบนกล้อง GPU (ภาพ GPU ต่างจาก CPU ≤1.5% ของพิกเซลที่ขอบวัตถุ) ; ภาพ 48×48 (ไม่ใช่ 128×128 ตามที่เสนอ – การอัปเดต CNN+GRU ที่ 128 px ช้าเกินไปใน JS ; ปรับได้ถึง 128 ในแผง) ; Obby ยังไม่มีโหมด first-person ; Reach ไม่มีสิ่งกีดขวางจึงยังไม่ได้ทดสอบ occlusion กับกำแพงจริง (depth buffer จัดการให้ในเรนเดอร์ GPU) ; ไม่มี adaptive rate อัตโนมัติ (ปรับขนาดภาพเอง) ; ใน Chromium แบบ SwiftShader ความเร็วช้ามาก (ไม่ใช่ตัวแทน GPU จริง)

## เทคโนโลยี
- **Three.js** (WebGL2, PBR, real-time shadow map 2048², bias/normalBias ปรับตามขนาดฉาก) · **Rapier** physics (fixed timestep 1/60 s)
- **Double DQN** เขียนเอง: MLP 17→64→64→14, Experience Replay (50k), ε-greedy, Target Network, Adam, Huber loss, gradient clipping
- **Web Worker**: Physics + AI รันแยกจาก UI/Rendering จึงไม่ทำให้ภาพกระตุก · จำกัดงานเทรนต่อเฟรมด้วย time budget
- **Observation (17 ค่า, normalize แล้ว)**: ตำแหน่ง/ทิศ/ระยะไปเป้า, ความเร็ว, แตะพื้น, ระยะถึงขอบ, ออกนอกสนาม ฯลฯ
- **Action (14)**: หยุด, 8 ทิศ, กระโดด, กระโดด+เคลื่อนที่ — ควบคุมผ่านแรงกระตุ้น (impulse) เท่านั้น ไม่เซ็ตตำแหน่งตรง ๆ
- **Reward**: ระยะที่ลดลงต่อสเต็ป (วนซ้ำไม่ได้คะแนน) + สำเร็จ +10 · ตกสนาม −5 · โทษเวลา/กระโดดเล็กน้อย
- ลดคุณภาพกราฟิกอัตโนมัติเมื่อเครื่องช้า (Auto/High/Medium/Low)

## โครงสร้าง (`src/`)
- `core/nn` · `core/rl` — Neural network, Double DQN, replay buffer
- `core/physics` · `core/agent` — Rapier wrapper, BallAgent
- `core/action` · `core/task` · `core/tasks/reachSquare` — Action system, Task/Environment interface, Environment + Observation + Reward ของแต่ละภารกิจ
- `core/training` — Training Manager, stats, import/export
- `worker/` — Simulation worker · `render/` — Three.js scene/camera/quality · `ui/` — แผงควบคุม กราฟ หน้าต่างเลือกภารกิจ · `persist/` — IndexedDB

### เพิ่มภารกิจใหม่
สร้างโฟลเดอร์ใน `core/tasks/<ชื่อ>/` (Environment, Observation, Reward, `info.ts`) แล้วลงทะเบียนใน `tasks/catalog.ts` (UI) และ `tasks/registry.ts` (worker) — หน้าต่างเลือกภารกิจจะแสดงให้อัตโนมัติ

## พัฒนาเอง
```bash
npm install
npm run dev          # dev server
npm run build        # -> dist/index.html (ไฟล์เดียว)
npm run test:learn   # เทรนแบบ headless แล้วตรวจว่าเรียนรู้จริง
```
เปิดผ่าน http(s) เท่านั้น (IndexedDB/Worker ไม่ทำงานบน `file://`)
