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
