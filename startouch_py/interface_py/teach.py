#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trajectory Teaching & Playback (6-DOF)
- Realtime gravity compensation (non-blocking, called every loop)
- Record (q, dq) at a fixed sample rate
- Replay using set_joint_raw with precise time alignment
- Save/Load CSV with headers

Keys:
  r : start/stop recording
  m : mark one sample immediately
  p : replay with original timestamps (set_joint_raw)
  P : replay uniformly re-timed (total duration = playback_tf)
  s : save CSV (trajectory.csv)
  l : load CSV (trajectory.csv)
  c : clear in-memory trajectory
  h : go home
  q : quit
"""

from __future__ import annotations

import os
import csv
import sys
import tty
import time
import signal
import termios
import select
from typing import List, Tuple, Optional
import numpy as np

from startouchclass import SingleArm  # 你的底层封装


# =============== 配置 ===============

DEFAULT_NUM_JOINTS = 6          # 默认 6 关节
KEY_READ_SLEEP_SEC = 0.002      # 主循环轻微 sleep，降低 CPU 占用
MIN_REPLAY_STEP_SEC = 0.002     # 回放等待时的最小 sleep 粒度
MIN_SEG_TF_SEC = 0.02           # 均匀回放的最小段时长，避免过短导致 jitter


# =============== 工具类：非阻塞键盘读取 ===============

class NonBlockingKeyReader:
    """Context manager 将 stdin 置为 cbreak，支持非阻塞 getch。"""
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self.fd)

    def __enter__(self) -> "NonBlockingKeyReader":
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old_settings)

    def getch(self, timeout: float = 0.0) -> Optional[str]:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            ch = sys.stdin.read(1)
            return ch
        return None


# =============== 主逻辑类 ===============

class TrajectoryTeacher:
    """
    轨迹示教/回放控制台（6 关节版本）

    Attributes
    ----------
    arm : SingleArm
        底层机械臂封装（需提供 gravity_compensation()、set_joint_raw() 等）
    num_joints : int
        关节个数（默认 6）
    sample_dt : float
        录制采样周期（秒）
    playback_dt : float
        均匀回放的控制步长（秒），仅用作时间片/等待粒度
    traj : List[Tuple[float, np.ndarray, np.ndarray]]
        轨迹缓存 [(t, q, dq)]，t 为相对时间（秒）
    """

    def __init__(
        self,
        can_interface: str = "can0",
        enable_fd: bool = False,
        sample_hz: float = 200.0,       # 建议较高采样频率，回放更顺滑
        playback_ctrl_hz: float = 350.0 # 均匀回放节拍（sleep 粒度）
    ) -> None:
        self.arm = SingleArm(can_interface_=can_interface, enable_fd_=enable_fd)

        # 如需自动探测关节数，可改为：len(self.arm.get_joint_positions())
        self.num_joints: int = DEFAULT_NUM_JOINTS

        self.sample_dt: float = 1.0 / float(sample_hz)
        self.playback_dt: float = 1.0 / float(playback_ctrl_hz)

        self.recording: bool = False
        self.traj: List[Tuple[float, np.ndarray, np.ndarray]] = []

        self._t0: Optional[float] = None
        self._last_sample_time: Optional[float] = None

        self.playback_tf: float = 5.0   # 均匀回放总时长默认值

        # 优雅退出
        signal.signal(signal.SIGINT, self._on_sigint)

    # ---------- 实用方法 ----------

    def _now(self) -> float:
        """单调时钟，适合做相对时间计时。"""
        return time.perf_counter()

    def _read_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取 (q, dq)，返回 np.float64 的一维数组。
        如底层 dq 读失败，可退化为 zeros（也可改为有限差分）。
        """
        q = np.asarray(self.arm.get_joint_positions(), dtype=np.float64).reshape(-1)
        if q.size != self.num_joints:
            raise RuntimeError(f"Expected {self.num_joints} joints, got {q.size}")

        try:
            dq = np.asarray(self.arm.get_joint_velocities(), dtype=np.float64).reshape(-1)
            if dq.size != self.num_joints:
                # 长度不符时退化为 0
                dq = np.zeros(self.num_joints, dtype=np.float64)
        except Exception:
            dq = np.zeros(self.num_joints, dtype=np.float64)

        return q, dq

    # ---------- 信号处理 ----------

    def _on_sigint(self, *_):
        print("\n[!] Ctrl+C 捕获，准备回零并退出…")
        try:
            self.arm.set_joint([0.0] * self.num_joints, tf=2.0)
        except Exception:
            pass
        time.sleep(0.1)
        sys.exit(0)

    # ---------- 基本动作 ----------

    def go_home(self) -> None:
        self.arm.go_home()

    # ---------- 录制 ----------

    def start_record(self) -> None:
        if self.recording:
            print("[Record] 已在录制中")
            return
        self.recording = True
        self.traj.clear()
        self._t0 = self._now()
        self._last_sample_time = None
        print("[Record] 开始录制")

    def stop_record(self) -> None:
        if not self.recording:
            print("[Record] 未在录制状态")
            return
        self.recording = False
        n = len(self.traj)
        dur = self.traj[-1][0] if n > 0 else 0.0
        print(f"[Record] 停止录制，共 {n} 帧，用时 {dur:.3f}s")

    def mark_once(self) -> None:
        if self._t0 is None:
            self._t0 = self._now()
        t = self._now() - self._t0
        q, dq = self._read_state()
        self.traj.append((t, q, dq))
        print(f"[Mark] t={t:.3f}s, q={np.round(q, 4)}")

    def _record_tick(self) -> None:
        """在主循环中周期性调用，满足采样周期则采样一帧。"""
        if not self.recording:
            return

        now = self._now()
        if (self._last_sample_time is None) or (now - self._last_sample_time >= self.sample_dt):
            if self._t0 is None:
                self._t0 = now
            t = now - self._t0
            q, dq = self._read_state()
            self.traj.append((t, q, dq))
            self._last_sample_time = now

    # ---------- 回放（按原时间戳） ----------

    def replay_time_stamped_raw(self) -> None:
        """
        使用 set_joint_raw(q, dq) 按采样时间戳回放。
        会尽可能对齐每一帧的相对时间，减少抖动。
        """
        if len(self.traj) < 2:
            print("[Replay] 轨迹不足 2 帧，无法回放")
            return

        print("[Replay] 按原时间戳 + set_joint_raw 回放开始")

        # 先到起始位（给一点 settle 时间）
        _, q0, dq0 = self.traj[0]
        self.arm.set_joint_raw(q0.tolist(), dq0.tolist())
        time.sleep(0.1)

        t_start = self._now()

        for t_rel, q, dq in self.traj:
            # 精确等待到该帧
            while True:
                elapsed = self._now() - t_start
                remain = t_rel - elapsed
                if remain <= 0:
                    break
                # 用较小粒度等待，避免过度忙等
                time.sleep(min(remain, MIN_REPLAY_STEP_SEC))

            self.arm.set_joint_raw(q.tolist(), dq.tolist())

        print("[Replay] 完成")

    # ---------- 回放（等时缩放） ----------

    def replay_uniform(self, total_tf: Optional[float] = None) -> None:
        """
        等速回放：将整段轨迹重采样为固定总时长 total_tf。
        对 (q, dq) 分别做线性插值。
        """
        if len(self.traj) < 2:
            print("[Replay] 轨迹不足 2 帧，无法回放")
            return

        if total_tf is None:
            total_tf = self.playback_tf

        ts = np.array([t for t, _, _ in self.traj], dtype=np.float64)
        qs = np.stack([q for _, q, _ in self.traj], axis=0)      # [N, J]
        dqs = np.stack([dq for _, _, dq in self.traj], axis=0)   # [N, J]

        t_span = float(ts[-1] - ts[0])
        if t_span <= 1e-9:
            print("[Replay] 时间跨度过小，取消")
            return

        N = max(20, int(total_tf / self.playback_dt))
        t_new = np.linspace(ts[0], ts[-1], N, dtype=np.float64)

        # 逐关节线性插值
        q_new = np.empty((N, self.num_joints), dtype=np.float64)
        dq_new = np.empty((N, self.num_joints), dtype=np.float64)
        for j in range(self.num_joints):
            q_new[:, j] = np.interp(t_new, ts, qs[:, j])
            dq_new[:, j] = np.interp(t_new, ts, dqs[:, j])

        # 先到起点
        self.arm.set_joint_raw(q_new[0].tolist(), dq_new[0].tolist())
        time.sleep(0.1)

        # 均匀小步回放
        small_tf = float(max(MIN_SEG_TF_SEC, total_tf / max(1, N - 1)))
        t0 = self._now()
        for i in range(1, N):
            # 对齐时间（避免累积漂移）
            target_time = (i * small_tf)
            while True:
                elapsed = self._now() - t0
                remain = target_time - elapsed
                if remain <= 0:
                    break
                time.sleep(min(remain, MIN_REPLAY_STEP_SEC))

            self.arm.set_joint_raw(q_new[i].tolist(), dq_new[i].tolist())

        print("[Replay] 完成")

    # ---------- CSV I/O ----------

    def save_csv(self, path: str = "trajectory1.csv") -> None:
        if not self.traj:
            print("[Save] 无数据")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        header = (["t"]
                  + [f"q{i+1}" for i in range(self.num_joints)]
                  + [f"dq{i+1}" for i in range(self.num_joints)])

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for t, q, dq in self.traj:
                row = [f"{t:.6f}"] + [f"{x:.9f}" for x in q] + [f"{x:.9f}" for x in dq]
                w.writerow(row)

        print(f"[Save] 已保存至 {path}（{len(self.traj)} 帧）")

    def load_csv(self, path: str = "trajectory1.csv") -> None:
        if not os.path.exists(path):
            print(f"[Load] 文件不存在: {path}")
            return

        traj: List[Tuple[float, np.ndarray, np.ndarray]] = []
        with open(path, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)

            # 兼容旧格式（只有 t + q1..qJ）
            # 或新格式（t + q1..qJ + dq1..dqJ）
            for row in r:
                if not row:
                    continue
                t = float(row[0])
                q_vals = np.array([float(x) for x in row[1:1+self.num_joints]], dtype=np.float64)
                if len(row) >= 1 + 2*self.num_joints:
                    dq_vals = np.array([float(x) for x in row[1+self.num_joints:1+2*self.num_joints]],
                                       dtype=np.float64)
                else:
                    dq_vals = np.zeros(self.num_joints, dtype=np.float64)
                traj.append((t, q_vals, dq_vals))

        if len(traj) < 2:
            print("[Load] 帧数不足")
            return

        # 归一化时间到从 0 开始
        t0 = traj[0][0]
        self.traj = [(t - t0, q, dq) for (t, q, dq) in traj]
        print(f"[Load] 已加载 {len(self.traj)} 帧")

    # ---------- 主循环 ----------

    def run(self) -> None:
        print("=" * 60)
        print("  轨迹示教 & 回放 — 控制台（6-DOF）")
        print("=" * 60)
        print(__doc__)
        print("-" * 60)

        with NonBlockingKeyReader() as kr:
            while True:
                # 1) 实时单步重力补偿（非阻塞）
                try:
                    self.arm.gravity_compensation()
                except Exception as e:
                    # 若出现底层异常，不要卡死主循环
                    print(f"[Warn] gravity_compensation() 异常: {e}")

                # 2) 录制 tick
                self._record_tick()

                # 3) 处理按键（非阻塞）
                ch = kr.getch(timeout=0.0)
                if ch == 'r':
                    (self.start_record() if not self.recording else self.stop_record())
                elif ch == 'm':
                    self.mark_once()
                elif ch == 'p':
                    self.replay_time_stamped_raw()
                elif ch == 'P':
                    self.replay_uniform()
                elif ch == 's':
                    self.save_csv()
                elif ch == 'l':
                    self.load_csv()
                elif ch == 'c':
                    self.traj.clear()
                    print("[Clear] 已清空轨迹")
                elif ch == 'h':
                    self.go_home()
                elif ch == 'q':
                    print("[Exit] bye")
                    break

                time.sleep(KEY_READ_SLEEP_SEC)


# =============== Entrypoint ===============

if __name__ == "__main__":
    teacher = TrajectoryTeacher(
        can_interface="can0",
        enable_fd=False,
        sample_hz=200.0,      # 录制 200Hz，回放会更顺滑
        playback_ctrl_hz=350.0
    )
    teacher.run()
