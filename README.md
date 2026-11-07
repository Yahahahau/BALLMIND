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
