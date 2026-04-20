from pettingzoo.classic import go_v5
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from flask import Flask, render_template_string
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import base64, io, threading, os

# ── Configuration ──────────────────────────────────────────────
BOARD_SIZE     = 19
TOTAL_TIMESTEPS = 500_000
SAVE_EVERY     = 50_000
LOG_DIR        = "./logs/ppo/"
SAVE_DIR       = "./models/ppo/"
WEB_PORT       = 5001

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Shared state between training thread and web server ────────
shared = {
    'frame':      None,
    'episode':    0,
    'wins':       0,
    'timestep':   0,
    'win_rate':   0.0,
    'ep_rew':     0.0,
    'fps':        0,
    'done':       False,
}
lock = threading.Lock()

# ── Go environment wrapper ─────────────────────────────────────
class GoEnvWrapper(gym.Env):
    def __init__(self, board_size=19, komi=7.5):
        super().__init__()
        self.board_size  = board_size
        self.komi        = komi
        # rgb_array lets us capture board frames
        self.env = go_v5.env(board_size=board_size, komi=komi, render_mode='rgb_array')
        self.action_space = spaces.Discrete(board_size * board_size + 1)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(board_size * board_size * 17,),
            dtype=np.float32
        )
        self.episode_count = 0
        self.win_count     = 0
        self.last_frame    = None

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        obs, _, _, _, _ = self.env.last()
        return self._process_obs(obs), {}

    def _process_obs(self, obs):
        return obs['observation'].flatten().astype(np.float32)

    def _capture_frame(self):
        frame = self.env.render()
        if frame is None:
            return
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        self.last_frame = base64.b64encode(buf.getvalue()).decode()

    def step(self, action):
        obs, reward, term, trunc, _ = self.env.last()
        action_mask = obs['action_mask']

        # Redirect illegal moves to a random legal one
        if action_mask[action] == 0:
            legal = np.where(action_mask == 1)[0]
            action = int(np.random.choice(legal))

        # Black (our agent) moves
        self.env.step(int(action))

        # White (random opponent) moves immediately after
        if not term and not trunc:
            opp_obs, _, opp_term, opp_trunc, _ = self.env.last()
            if not opp_term and not opp_trunc:
                opp_mask   = opp_obs['action_mask']
                legal_opp  = np.where(opp_mask == 1)[0]
                self.env.step(int(np.random.choice(legal_opp)))

        # Capture board frame after every move
        self._capture_frame()

        next_obs, next_reward, next_term, next_trunc, _ = self.env.last()

        if next_term or next_trunc:
            self.episode_count += 1
            if next_reward > 0:
                self.win_count += 1

        return (
            self._process_obs(next_obs),
            float(next_reward),
            next_term,
            next_trunc,
            {}
        )

    def close(self):
        self.env.close()


# ── Live board callback ────────────────────────────────────────
# Runs every training step — pushes board frame + stats to shared state
class LiveBoardCallback(BaseCallback):
    def __init__(self, save_every, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_every = save_every
        self.save_dir   = save_dir
        self.last_save  = 0

    def _on_step(self):
        env = self.training_env.envs[0].env

        # Push latest frame and stats to shared state
        if env.last_frame:
            with lock:
                shared['frame']    = env.last_frame
                shared['episode']  = env.episode_count
                shared['wins']     = env.win_count
                shared['timestep'] = self.num_timesteps
                shared['win_rate'] = env.win_count / max(env.episode_count, 1)

        # Log win rate to TensorBoard
        if env.episode_count > 0:
            wr = env.win_count / env.episode_count
            self.logger.record('custom/win_rate', wr)
            self.logger.record('custom/total_episodes', env.episode_count)


        # Clean terminal output every 10 timesteps
        if self.num_timesteps % 1 == 0:
            wr = env.win_count / max(env.episode_count, 1)
            print(f"  Step {self.num_timesteps:>8,} | "
                  f"Episodes: {env.episode_count:>5,} | "
                  f"Win rate: {wr:.1%} | "
                  f"{'✓ BEATING BASELINE' if wr > 0.40 else '  below baseline'}")
            
        # Save checkpoint
        if self.num_timesteps - self.last_save >= self.save_every:
            path = os.path.join(
                self.save_dir,
                f"ppo_go_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            print(f"\n  ✓ Checkpoint saved: {path}")
            self.last_save = self.num_timesteps

        return True

    def _on_training_end(self):
        with lock:
            shared['done'] = True


# ── Flask web server ───────────────────────────────────────────
app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>PPO Training — Live Go Board</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background: #111;
      color: #ddd;
      font-family: -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      gap: 16px;
    }
    .header {
      display: flex;
      align-items: center;
      gap: 12px;
      width: 640px;
    }
    h1 { font-size: 16px; font-weight: 500; color: #fff; flex: 1; }
    .badge {
      font-size: 11px; padding: 3px 10px;
      border-radius: 12px; font-weight: 500;
    }
    .badge-training { background: #1D9E75; color: #fff; }
    .badge-done     { background: #555;    color: #aaa; }

    #board {
      width: 640px; height: 640px;
      border-radius: 8px;
      border: 1px solid #333;
      display: block;
      background: #222;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      width: 640px;
    }
    .stat {
      background: #1e1e1e;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .stat-label { font-size: 10px; color: #666; margin-bottom: 4px; }
    .stat-value { font-size: 20px; font-weight: 500; color: #fff; }
    .green  { color: #4caf50 !important; }
    .red    { color: #ef5350 !important; }
    .yellow { color: #ffb74d !important; }

    .progress-bar-bg {
      width: 640px; height: 6px;
      background: #222; border-radius: 3px; overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%; background: #1D9E75;
      border-radius: 3px;
      transition: width 0.5s ease;
    }

    .footer {
      font-size: 11px; color: #444;
      display: flex; gap: 20px;
    }
  </style>
</head>
<body>

<div class="header">
  <h1>PPO Training — Live 19×19 Go Board</h1>
  <span class="badge badge-training" id="status-badge">● TRAINING</span>
</div>

<div class="progress-bar-bg">
  <div class="progress-bar-fill" id="progress" style="width:0%"></div>
</div>

<img id="board" alt="Go board loading..."/>

<div class="stats">
  <div class="stat">
    <div class="stat-label">Timestep</div>
    <div class="stat-value" id="timestep">0</div>
  </div>
  <div class="stat">
    <div class="stat-label">Episode</div>
    <div class="stat-value" id="episode">0</div>
  </div>
  <div class="stat">
    <div class="stat-label">Win Rate</div>
    <div class="stat-value" id="winrate">—</div>
  </div>
  <div class="stat">
    <div class="stat-label">Wins</div>
    <div class="stat-value green" id="wins">0</div>
  </div>
  <div class="stat">
    <div class="stat-label">Baseline</div>
    <div class="stat-value yellow">40%</div>
  </div>
</div>

<div class="footer">
  <span>■ Black = PPO agent (learning)</span>
  <span>□ White = random opponent</span>
  <span>TensorBoard → localhost:6006</span>
  <span>Updates every 300ms</span>
</div>

<script>
  const TOTAL = {{ total_timesteps }};

  function update() {
    fetch('/state')
      .then(r => r.json())
      .then(d => {

        // board image
        if (d.frame) {
          document.getElementById('board').src =
            'data:image/png;base64,' + d.frame;
        }

        // stats
        document.getElementById('timestep').textContent =
          d.timestep.toLocaleString();
        document.getElementById('episode').textContent =
          d.episode.toLocaleString();
        document.getElementById('wins').textContent = d.wins;

        // win rate with colour
        const wrEl = document.getElementById('winrate');
        if (d.episode > 0) {
          const pct = (d.win_rate * 100).toFixed(1) + '%';
          wrEl.textContent = pct;
          wrEl.className = 'stat-value ' +
            (d.win_rate > 0.40 ? 'green' : d.win_rate > 0.30 ? 'yellow' : 'red');
        }

        // progress bar
        const pct = Math.min(d.timestep / TOTAL * 100, 100);
        document.getElementById('progress').style.width = pct + '%';

        // status badge
        if (d.done) {
          const b = document.getElementById('status-badge');
          b.textContent = '✓ DONE';
          b.className = 'badge badge-done';
        }
      });
  }

  setInterval(update, 300);
  update();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(PAGE, total_timesteps=TOTAL_TIMESTEPS)

@app.route('/state')
def get_state():
    with lock:
        return dict(shared)


# ── Training runs in background thread ────────────────────────
def run_training():
    print("\nSetting up environment...")
    env = GoEnvWrapper(board_size=BOARD_SIZE)
    check_env(env, warn=True)
    print("Environment ready!")

    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
    )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"TensorBoard:  tensorboard --logdir {LOG_DIR}")
    print(f"Live board:   http://localhost:{WEB_PORT}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=LiveBoardCallback(
            save_every=SAVE_EVERY,
            save_dir=SAVE_DIR
        ),
        tb_log_name="ppo_run_1"
    )

    # Save final model
    final = os.path.join(SAVE_DIR, "ppo_go_final")
    model.save(final)
    print(f"\nFinal model saved: {final}")

    # Final evaluation
    print("\nRunning final evaluation (50 games)...")
    wins = 0
    for _ in range(50):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
        if reward > 0:
            wins += 1

    print(f"\nFinal win rate: {wins/50:.1%} (baseline: 40%)")
    env.close()


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  PPO Live Training — Go 19x19")
    print("=" * 55)

    # Start training in background thread
    train_thread = threading.Thread(target=run_training, daemon=True)
    train_thread.start()

    print(f"\n  Live board:  http://localhost:{WEB_PORT}")
    print(f"  TensorBoard: run in separate terminal:")
    print(f"               tensorboard --logdir {LOG_DIR}")
    print(f"\n  Press Ctrl+C to stop\n")

    # Flask runs on main thread
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)