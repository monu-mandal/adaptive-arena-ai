import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// CONSTANTS & CONFIG
// ============================================================
const W = 800, H = 450;
const GROUND = H - 60;
const GRAVITY = 0.55;
const JUMP_V = -13;
const MOVE_SPEED = 4.5;
const PLAYER_W = 36, PLAYER_H = 52;
const ATTACK_RANGE = 70;
const ATTACK_DMG_BASE = 12;
const BLOCK_DMG_REDUCE = 0.25;
const MAX_HP = 100;
const AI_ACTIONS = ["idle","moveLeft","moveRight","jump","attack","block","dash"];
const STATE_SIZE = 12;
const ACTION_SIZE = AI_ACTIONS.length;
const LR = 0.008;
const GAMMA = 0.92;
const EPSILON_DECAY = 0.9995;
const EPSILON_MIN = 0.05;
const PERSONALITIES = ["Aggressive","Defensive","Adaptive","Berserker","Tactician"];
const PERSONALITY_COLORS = ["#ff4444","#44aaff","#44ff88","#ff8800","#cc44ff"];

// ============================================================
// SIMPLE NEURAL NETWORK (no external deps)
// ============================================================
function randn() { 
  let u = 0, v = 0;
  while(!u) u = Math.random();
  while(!v) v = Math.random();
  return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
}
function sigmoid(x) { return 1/(1+Math.exp(-x)); }
function relu(x) { return Math.max(0,x); }

class SimpleNN {
  constructor(sizes) {
    this.sizes = sizes;
    this.weights = [];
    this.biases = [];
    for(let i=0;i<sizes.length-1;i++){
      const rows = sizes[i+1], cols = sizes[i];
      this.weights.push(Array.from({length:rows},()=>Array.from({length:cols},()=>randn()*Math.sqrt(2/cols))));
      this.biases.push(Array.from({length:rows},()=>0));
    }
  }
  forward(input) {
    let x = [...input];
    for(let l=0;l<this.weights.length;l++){
      const nx = [];
      for(let i=0;i<this.weights[l].length;i++){
        let s = this.biases[l][i];
        for(let j=0;j<x.length;j++) s += this.weights[l][j<=this.weights[l][i].length-1?j:0]*x[j]||0;
        // fix: proper dot product
        s = this.biases[l][i];
        for(let j=0;j<this.weights[l][i].length;j++) s += (this.weights[l][i][j]||0)*(x[j]||0);
        nx.push(l===this.weights.length-1 ? s : relu(s));
      }
      x = nx;
    }
    // softmax on output
    const maxV = Math.max(...x);
    const exps = x.map(v=>Math.exp(v-maxV));
    const sum = exps.reduce((a,b)=>a+b,0.001);
    return exps.map(v=>v/sum);
  }
  update(state, actionIdx, tdError, lr) {
    // Simple gradient step: nudge weights toward reducing TD error
    const probs = this.forward(state);
    const factor = lr * tdError;
    // Only update last layer for speed
    const L = this.weights.length-1;
    for(let i=0;i<this.weights[L].length;i++){
      const grad = (i===actionIdx ? 1-probs[i] : -probs[i]) * factor;
      for(let j=0;j<this.weights[L][i].length;j++){
        const prevX = this._lastHidden ? (this._lastHidden[j]||0) : (state[j]||0);
        this.weights[L][i][j] += grad * prevX;
      }
      this.biases[L][i] += grad;
    }
  }
  clone() {
    const nn = new SimpleNN([...this.sizes]);
    nn.weights = this.weights.map(w=>w.map(row=>[...row]));
    nn.biases = this.biases.map(b=>[...b]);
    return nn;
  }
  serialize() {
    return {sizes:this.sizes, weights:this.weights, biases:this.biases};
  }
}

// ============================================================
// RL AGENT
// ============================================================
class RLAgent {
  constructor(level=1, personality=0) {
    this.qNet = new SimpleNN([STATE_SIZE, 24, 18, ACTION_SIZE]);
    this.targetNet = this.qNet.clone();
    this.epsilon = Math.max(EPSILON_MIN, 0.9 - (level-1)*0.1);
    this.lr = LR;
    this.gamma = GAMMA;
    this.memory = [];
    this.memMaxSize = 800;
    this.level = level;
    this.personality = personality;
    this.totalReward = 0;
    this.episodeCount = 0;
    this.updateCount = 0;
    this.winRate = 0;
    this.wins = 0;
    this.losses = 0;
    this.rewardHistory = [];
    this.lastState = null;
    this.lastAction = null;
    this.stepReward = 0;
    // personality bias weights
    this.actionBias = this._buildBias(personality);
  }
  _buildBias(p) {
    // 0=aggressive,1=defensive,2=adaptive,3=berserker,4=tactician
    const b = new Array(ACTION_SIZE).fill(0);
    if(p===0){b[4]=0.3;b[2]=0.15;} // attack+moveRight
    if(p===1){b[5]=0.3;b[0]=0.1;}  // block+idle
    if(p===2){b[4]=0.1;b[5]=0.1;}  // balanced
    if(p===3){b[4]=0.5;b[3]=0.2;}  // attack+jump
    if(p===4){b[3]=0.2;b[6]=0.2;}  // jump+dash
    return b;
  }
  getState(ai, player, frameCount) {
    return [
      (ai.x - player.x) / W,
      (ai.y - player.y) / H,
      ai.vx / 10,
      ai.vy / 10,
      player.vx / 10,
      player.vy / 10,
      ai.hp / MAX_HP,
      player.hp / MAX_HP,
      ai.isAttacking ? 1 : 0,
      player.isAttacking ? 1 : 0,
      ai.onGround ? 1 : 0,
      (frameCount % 60) / 60,
    ];
  }
  selectAction(state) {
    if(Math.random() < this.epsilon) {
      // biased random by personality
      const weights = this.actionBias.map(b=>Math.max(0.05,0.1+b));
      const sum = weights.reduce((a,v)=>a+v,0);
      let r = Math.random()*sum;
      for(let i=0;i<weights.length;i++){r-=weights[i];if(r<=0)return i;}
      return Math.floor(Math.random()*ACTION_SIZE);
    }
    const probs = this.qNet.forward(state);
    // apply personality bias
    const biased = probs.map((p,i)=>p+this.actionBias[i]*0.2);
    return biased.indexOf(Math.max(...biased));
  }
  store(s,a,r,ns,done) {
    this.memory.push({s,a,r,ns,done});
    if(this.memory.length > this.memMaxSize) this.memory.shift();
    this.stepReward += r;
  }
  train() {
    if(this.memory.length < 32) return;
    const batch = [];
    for(let i=0;i<32;i++) batch.push(this.memory[Math.floor(Math.random()*this.memory.length)]);
    let totalLoss = 0;
    for(const {s,a,r,ns,done} of batch){
      const nextProbs = this.targetNet.forward(ns);
      const maxNext = Math.max(...nextProbs);
      const target = r + (done ? 0 : this.gamma*maxNext);
      const curProbs = this.qNet.forward(s);
      const tdError = target - curProbs[a];
      this.qNet.update(s, a, tdError, this.lr);
      totalLoss += Math.abs(tdError);
    }
    this.updateCount++;
    if(this.updateCount % 50 === 0) this.targetNet = this.qNet.clone();
    this.epsilon = Math.max(EPSILON_MIN, this.epsilon * EPSILON_DECAY);
    return totalLoss/32;
  }
  onEpisodeEnd(won) {
    this.episodeCount++;
    if(won) this.wins++; else this.losses++;
    this.winRate = this.wins / Math.max(1, this.wins+this.losses);
    this.totalReward += this.stepReward;
    this.rewardHistory.push(Math.round(this.stepReward*10)/10);
    if(this.rewardHistory.length > 20) this.rewardHistory.shift();
    this.stepReward = 0;
  }
}

// ============================================================
// GAME ENTITY
// ============================================================
function makeEntity(x, y, isPlayer) {
  return {
    x, y, vx: 0, vy: 0,
    w: PLAYER_W, h: PLAYER_H,
    hp: MAX_HP, maxHp: MAX_HP,
    onGround: false,
    isAttacking: false, attackTimer: 0, attackCooldown: 0,
    isBlocking: false, blockTimer: 0,
    isDashing: false, dashTimer: 0, dashCooldown: 0,
    facingRight: isPlayer,
    hitFlash: 0, deathAnim: 0,
    comboCount: 0, comboTimer: 0,
    isPlayer,
    jumpCount: 0,
  };
}

function applyPhysics(e) {
  if(!e.onGround) e.vy += GRAVITY;
  e.x += e.vx;
  e.y += e.vy;
  if(e.y + e.h >= GROUND) {
    e.y = GROUND - e.h;
    e.vy = 0;
    e.onGround = true;
    e.jumpCount = 0;
  } else {
    e.onGround = false;
  }
  e.x = Math.max(0, Math.min(W - e.w, e.x));
  // friction
  e.vx *= 0.82;
  if(Math.abs(e.vx) < 0.1) e.vx = 0;
  // timers
  if(e.attackTimer > 0) e.attackTimer--;
  if(e.attackTimer === 0) e.isAttacking = false;
  if(e.attackCooldown > 0) e.attackCooldown--;
  if(e.blockTimer > 0) e.blockTimer--;
  if(e.blockTimer === 0) e.isBlocking = false;
  if(e.dashTimer > 0) e.dashTimer--;
  if(e.dashTimer === 0) e.isDashing = false;
  if(e.dashCooldown > 0) e.dashCooldown--;
  if(e.hitFlash > 0) e.hitFlash--;
  if(e.comboTimer > 0) e.comboTimer--;
  if(e.comboTimer === 0) e.comboCount = 0;
}

function checkAttackHit(attacker, defender) {
  if(!attacker.isAttacking || attacker.attackTimer < 8) return 0;
  const dx = Math.abs((attacker.x+attacker.w/2) - (defender.x+defender.w/2));
  if(dx > ATTACK_RANGE) return 0;
  // check facing
  const facing = attacker.facingRight ? 1 : -1;
  const dir = (defender.x - attacker.x) * facing;
  if(dir < -10) return 0;
  let dmg = ATTACK_DMG_BASE + (attacker.comboCount > 0 ? attacker.comboCount*2 : 0);
  if(defender.isBlocking) dmg = Math.ceil(dmg * BLOCK_DMG_REDUCE);
  return dmg;
}

// ============================================================
// MAIN GAME COMPONENT
// ============================================================
export default function AdaptiveArena() {
  const canvasRef = useRef(null);
  const gameStateRef = useRef(null);
  const agentRef = useRef(null);
  const keysRef = useRef({});
  const animFrameRef = useRef(null);
  const frameCountRef = useRef(0);

  const [uiState, setUiState] = useState({
    screen: "menu", // menu | game | gameover | levelup
    level: 1,
    personality: 0,
    playerHp: MAX_HP,
    aiHp: MAX_HP,
    playerScore: 0,
    aiScore: 0,
    round: 1,
    epsilon: 0.9,
    winRate: 0,
    episodeCount: 0,
    rewardHistory: [],
    lastResult: null,
    combo: 0,
    aiCombo: 0,
    aiAction: "idle",
    showStats: true,
    particles: [],
    roundTimer: 0,
  });

  const particlesRef = useRef([]);

  function spawnParticles(x, y, color, count=6) {
    for(let i=0;i<count;i++){
      particlesRef.current.push({
        x, y,
        vx: (Math.random()-0.5)*6,
        vy: (Math.random()-0.6)*6,
        life: 30+Math.random()*20,
        maxLife: 50,
        color,
        r: 2+Math.random()*4,
      });
    }
  }

  function initGame(level, personality) {
    const player = makeEntity(120, GROUND - PLAYER_H, true);
    const ai = makeEntity(W - 160, GROUND - PLAYER_H, false);
    if(!agentRef.current || agentRef.current.level !== level || agentRef.current.personality !== personality){
      agentRef.current = new RLAgent(level, personality);
    }
    gameStateRef.current = {
      player, ai,
      roundActive: true,
      roundTimer: 90 * 60,
      hitConnected: { player: false, ai: false },
    };
    frameCountRef.current = 0;
    particlesRef.current = [];
  }

  const endRound = useCallback((playerWon) => {
    const gs = gameStateRef.current;
    const agent = agentRef.current;
    if(!gs || !gs.roundActive) return;
    gs.roundActive = false;
    agent.onEpisodeEnd(!playerWon);

    setUiState(prev => {
      const newPlayerScore = prev.playerScore + (playerWon ? 1 : 0);
      const newAiScore = prev.aiScore + (playerWon ? 0 : 1);
      return {
        ...prev,
        screen: "gameover",
        lastResult: playerWon ? "win" : "lose",
        playerScore: newPlayerScore,
        aiScore: newAiScore,
        epsilon: Math.round(agent.epsilon*1000)/1000,
        winRate: Math.round(agent.winRate*100),
        episodeCount: agent.episodeCount,
        rewardHistory: [...agent.rewardHistory],
      };
    });
  }, []);

  const gameLoop = useCallback(() => {
    const gs = gameStateRef.current;
    const agent = agentRef.current;
    if(!gs || !gs.roundActive) return;

    const {player, ai} = gs;
    const keys = keysRef.current;
    frameCountRef.current++;
    const fc = frameCountRef.current;
    gs.roundTimer--;

    // --- PLAYER INPUT ---
    if(keys["ArrowLeft"]||keys["a"]) { player.vx = -MOVE_SPEED; player.facingRight=false; }
    if(keys["ArrowRight"]||keys["d"]) { player.vx = MOVE_SPEED; player.facingRight=true; }
    if((keys["ArrowUp"]||keys["w"]||keys[" "]) && player.jumpCount < 2) {
      player.vy = JUMP_V * (player.jumpCount===1?0.8:1);
      player.jumpCount++;
      keys["ArrowUp"]=false; keys["w"]=false; keys[" "]=false;
    }
    if((keys["z"]||keys["j"]) && player.attackCooldown===0) {
      player.isAttacking=true; player.attackTimer=12; player.attackCooldown=18;
      if(player.comboTimer>0) player.comboCount=Math.min(3,player.comboCount+1);
      else player.comboCount=1;
      player.comboTimer=40;
    }
    if(keys["x"]||keys["k"]) { player.isBlocking=true; player.blockTimer=5; }
    if((keys["c"]||keys["l"]) && player.dashCooldown===0) {
      player.vx = player.facingRight ? 11 : -11;
      player.isDashing=true; player.dashTimer=8; player.dashCooldown=45;
    }

    // --- AI DECISION ---
    const state = agent.getState(ai, player, fc);
    let actionIdx = 0;
    if(fc % 6 === 0) {  // AI thinks every 6 frames
      actionIdx = agent.selectAction(state);
      agent.lastState = state;
      agent.lastAction = actionIdx;
    } else {
      actionIdx = agent.lastAction ?? 0;
    }
    const aiAction = AI_ACTIONS[actionIdx];

    // Execute AI action
    if(aiAction==="moveLeft") { ai.vx=-MOVE_SPEED*0.9; ai.facingRight=false; }
    if(aiAction==="moveRight") { ai.vx=MOVE_SPEED*0.9; ai.facingRight=true; }
    if(aiAction==="jump" && ai.jumpCount<2) {
      ai.vy=JUMP_V*(ai.jumpCount===1?0.8:1); ai.jumpCount++;
    }
    if(aiAction==="attack" && ai.attackCooldown===0) {
      ai.isAttacking=true; ai.attackTimer=12; ai.attackCooldown=20;
      if(ai.comboTimer>0) ai.comboCount=Math.min(3,ai.comboCount+1);
      else ai.comboCount=1;
      ai.comboTimer=40;
    }
    if(aiAction==="block") { ai.isBlocking=true; ai.blockTimer=5; }
    if(aiAction==="dash" && ai.dashCooldown===0) {
      ai.vx=ai.facingRight?10:-10; ai.isDashing=true; ai.dashTimer=8; ai.dashCooldown=50;
    }
    // AI always faces player
    ai.facingRight = ai.x < player.x;

    // --- PHYSICS ---
    applyPhysics(player);
    applyPhysics(ai);

    // --- COMBAT ---
    let reward = 0;
    gs.hitConnected = {player:false, ai:false};

    const pDmg = checkAttackHit(player, ai);
    if(pDmg > 0) {
      ai.hp = Math.max(0, ai.hp - pDmg);
      ai.hitFlash = 8;
      gs.hitConnected.player = true;
      reward -= pDmg * 0.3;
      spawnParticles(ai.x+ai.w/2, ai.y+ai.h/2, "#ff4444", pDmg>10?10:6);
      player.attackTimer = -1; player.isAttacking = false; // reset so no double hit
    }
    const aDmg = checkAttackHit(ai, player);
    if(aDmg > 0) {
      player.hp = Math.max(0, player.hp - aDmg);
      player.hitFlash = 8;
      gs.hitConnected.ai = true;
      reward += aDmg * 0.3;
      spawnParticles(player.x+player.w/2, player.y+player.h/2, "#ff8800", aDmg>10?10:6);
      ai.attackTimer = -1; ai.isAttacking = false;
    }

    // Shaping rewards
    const dist = Math.abs(ai.x - player.x);
    if(dist < 100) reward += 0.02;
    if(ai.isBlocking && player.isAttacking) reward += 0.1;
    reward += (ai.hp - player.hp) / MAX_HP * 0.01;

    // Store transition
    if(agent.lastState && agent.lastAction !== null && fc % 6 === 0){
      const ns = agent.getState(ai, player, fc);
      agent.store(agent.lastState, agent.lastAction, reward, ns, false);
      agent.train();
    }

    // Update particles
    particlesRef.current = particlesRef.current
      .filter(p=>p.life>0)
      .map(p=>({...p, x:p.x+p.vx, y:p.y+p.vy+0.3, life:p.life-1, vx:p.vx*0.9}));

    // --- CHECK WIN ---
    if(player.hp <= 0) { endRound(false); return; }
    if(ai.hp <= 0) { endRound(true); return; }
    if(gs.roundTimer <= 0) { endRound(player.hp > ai.hp); return; }

    // Update UI occasionally
    if(fc % 8 === 0) {
      setUiState(prev => ({
        ...prev,
        playerHp: Math.round(player.hp),
        aiHp: Math.round(ai.hp),
        combo: player.comboCount,
        aiCombo: ai.comboCount,
        aiAction: aiAction,
        roundTimer: Math.ceil(gs.roundTimer/60),
        epsilon: Math.round(agent.epsilon*1000)/1000,
        winRate: Math.round(agent.winRate*100),
        episodeCount: agent.episodeCount,
        rewardHistory: fc%48===0 ? [...agent.rewardHistory] : prev.rewardHistory,
      }));
    }
  }, [endRound]);

  // RENDER
  const renderGame = useCallback(() => {
    const canvas = canvasRef.current;
    if(!canvas) return;
    const ctx = canvas.getContext("2d");
    const gs = gameStateRef.current;
    if(!gs) return;
    const {player, ai} = gs;

    // Background
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0,0,W,H);

    // Arena grid
    ctx.strokeStyle = "rgba(40,60,120,0.15)";
    ctx.lineWidth = 1;
    for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}

    // Ground
    const gGrad = ctx.createLinearGradient(0,GROUND,0,H);
    gGrad.addColorStop(0,"#1a1a2e");
    gGrad.addColorStop(1,"#0d0d1a");
    ctx.fillStyle = gGrad;
    ctx.fillRect(0,GROUND,W,H-GROUND);
    ctx.strokeStyle = "#334";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0,GROUND); ctx.lineTo(W,GROUND); ctx.stroke();
    // ground accent line
    ctx.strokeStyle = "rgba(80,120,255,0.3)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0,GROUND+1); ctx.lineTo(W,GROUND+1); ctx.stroke();

    // Shadow under entities
    [player, ai].forEach(e=>{
      if(e.hp<=0) return;
      ctx.beginPath();
      ctx.ellipse(e.x+e.w/2, GROUND+2, e.w*0.5, 5, 0, 0, Math.PI*2);
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fill();
    });

    // Particles
    for(const p of particlesRef.current){
      const alpha = p.life/p.maxLife;
      ctx.beginPath();
      ctx.arc(p.x,p.y,p.r*alpha,0,Math.PI*2);
      ctx.fillStyle = p.color+(Math.floor(alpha*255).toString(16).padStart(2,"0"));
      ctx.fill();
    }

    // Draw entity function
    function drawEntity(e, isPlayer) {
      if(e.hp <= 0) return;
      const cx = e.x + e.w/2;
      const baseColor = isPlayer ? "#4af" : PERSONALITY_COLORS[agentRef.current?.personality||0];
      const darkColor = isPlayer ? "#08f" : "#cc2200";

      ctx.save();
      // Hit flash
      if(e.hitFlash > 0) {
        ctx.shadowBlur = 20;
        ctx.shadowColor = "#fff";
      }

      // Body glow
      if(e.isAttacking) {
        ctx.shadowBlur = 15;
        ctx.shadowColor = isPlayer ? "#4affff" : "#ff4400";
      }
      if(e.isDashing) {
        ctx.shadowBlur = 20;
        ctx.shadowColor = isPlayer ? "#00ff88" : "#ffaa00";
      }

      // Legs
      const legAnim = e.vx !== 0 && e.onGround ? Math.sin(Date.now()*0.02)*8 : 0;
      // left leg
      ctx.fillStyle = isPlayer ? "#2288cc" : "#882200";
      ctx.fillRect(e.x+3, e.y+e.h*0.55, e.w*0.35, e.h*0.45);
      // right leg
      ctx.fillRect(e.x+e.w*0.55, e.y+e.h*0.55, e.w*0.35, e.h*0.45);

      // Torso
      const torsoGrad = ctx.createLinearGradient(e.x,e.y,e.x+e.w,e.y+e.h);
      torsoGrad.addColorStop(0, e.hitFlash>0?"#ffffff":baseColor);
      torsoGrad.addColorStop(1, e.hitFlash>0?"#aaaaaa":darkColor);
      ctx.fillStyle = torsoGrad;
      ctx.beginPath();
      ctx.roundRect(e.x+2, e.y+e.h*0.18, e.w-4, e.h*0.42, 4);
      ctx.fill();

      // Block shield
      if(e.isBlocking){
        ctx.fillStyle = isPlayer?"rgba(80,180,255,0.4)":"rgba(255,180,80,0.4)";
        ctx.beginPath();
        const shieldX = e.facingRight ? e.x+e.w-4 : e.x-8;
        ctx.roundRect(shieldX, e.y+4, 12, e.h*0.7, 3);
        ctx.fill();
        ctx.strokeStyle = isPlayer?"#4af":"#fa0";
        ctx.lineWidth=2; ctx.stroke();
      }

      // Head
      ctx.fillStyle = e.hitFlash>0?"#ffffff":baseColor;
      ctx.beginPath();
      ctx.arc(cx, e.y+e.h*0.12, e.w*0.38, 0, Math.PI*2);
      ctx.fill();

      // Eye
      ctx.fillStyle = "#000";
      const eyeX = e.facingRight ? cx+4 : cx-4;
      ctx.beginPath();
      ctx.arc(eyeX, e.y+e.h*0.10, 3, 0, Math.PI*2);
      ctx.fill();
      ctx.fillStyle = isPlayer?"#0ff":"#f80";
      ctx.beginPath();
      ctx.arc(eyeX, e.y+e.h*0.10, 1.5, 0, Math.PI*2);
      ctx.fill();

      // Attack hitbox glow
      if(e.isAttacking && e.attackTimer > 4) {
        const atkX = e.facingRight ? e.x+e.w : e.x-ATTACK_RANGE;
        ctx.fillStyle = isPlayer?"rgba(80,255,255,0.15)":"rgba(255,100,0,0.15)";
        ctx.strokeStyle = isPlayer?"rgba(80,255,255,0.4)":"rgba(255,100,0,0.4)";
        ctx.lineWidth=1;
        ctx.beginPath();
        ctx.roundRect(atkX, e.y, ATTACK_RANGE, e.h, 4);
        ctx.fill(); ctx.stroke();
      }

      // Dash trail
      if(e.isDashing) {
        ctx.fillStyle = isPlayer?"rgba(0,255,136,0.1)":"rgba(255,170,0,0.1)";
        ctx.fillRect(e.x - e.vx*3, e.y, e.w, e.h);
        ctx.fillRect(e.x - e.vx*6, e.y, e.w*0.8, e.h*0.8);
      }

      ctx.restore();
    }

    drawEntity(player, true);
    drawEntity(ai, false);

    // Arena name watermark
    ctx.font = "bold 11px monospace";
    ctx.fillStyle = "rgba(100,100,160,0.2)";
    ctx.textAlign = "center";
    ctx.fillText("ADAPTIVE ARENA AI", W/2, H-8);

  }, []);

  // MAIN LOOP
  useEffect(() => {
    if(uiState.screen !== "game") return;
    let running = true;
    const loop = () => {
      if(!running) return;
      gameLoop();
      renderGame();
      animFrameRef.current = requestAnimationFrame(loop);
    };
    animFrameRef.current = requestAnimationFrame(loop);
    return () => { running = false; cancelAnimationFrame(animFrameRef.current); };
  }, [uiState.screen, gameLoop, renderGame]);

  // KEYBOARD
  useEffect(() => {
    const down = e => { keysRef.current[e.key] = true; e.preventDefault(); };
    const up = e => { keysRef.current[e.key] = false; };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  function startGame(level, personality) {
    initGame(level, personality);
    setUiState(prev => ({
      ...prev,
      screen: "game",
      level, personality,
      playerHp: MAX_HP, aiHp: MAX_HP,
      combo: 0, aiCombo: 0,
      roundTimer: 90,
    }));
  }

  function continueGame() {
    const agent = agentRef.current;
    const newLevel = uiState.lastResult === "win" ? Math.min(10, uiState.level + 1) : uiState.level;
    const newPersonality = newLevel > uiState.level ? (uiState.personality + (Math.random()>0.5?1:0)) % PERSONALITIES.length : uiState.personality;
    // Keep agent, just reinit entities
    const player = makeEntity(120, GROUND - PLAYER_H, true);
    const ai2 = makeEntity(W - 160, GROUND - PLAYER_H, false);
    if(newLevel !== uiState.level && newLevel > uiState.level) {
      // Level up: boost agent
      agent.level = newLevel;
      agent.epsilon = Math.max(EPSILON_MIN, agent.epsilon - 0.05);
      agent.personality = newPersonality;
      agent.actionBias = agent._buildBias(newPersonality);
    }
    gameStateRef.current = { player, ai: ai2, roundActive: true, roundTimer: 90*60, hitConnected:{player:false,ai:false} };
    frameCountRef.current = 0;
    particlesRef.current = [];
    setUiState(prev => ({
      ...prev,
      screen: "game",
      level: newLevel,
      personality: newPersonality,
      playerHp: MAX_HP, aiHp: MAX_HP,
      combo: 0, aiCombo: 0, roundTimer: 90,
    }));
  }

  // ---- UI ----
  const pct = v => `${Math.max(0,Math.min(100,v))}%`;
  const rewardHistory = uiState.rewardHistory;
  const rewardMax = rewardHistory.length ? Math.max(...rewardHistory.map(Math.abs), 1) : 1;

  // MENU SCREEN
  if(uiState.screen === "menu") {
    return (
      <div style={{background:"#07070f",width:"100vw",height:"100vh",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",fontFamily:"'Courier New',monospace",color:"#ccc",overflow:"hidden",position:"relative"}}>
        <div style={{position:"absolute",inset:0,background:"radial-gradient(ellipse at 50% 60%, rgba(40,60,200,0.12) 0%, transparent 70%)",pointerEvents:"none"}}/>
        {/* Scanlines */}
        <div style={{position:"absolute",inset:0,background:"repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.15) 2px,rgba(0,0,0,0.15) 4px)",pointerEvents:"none",opacity:0.5}}/>

        <div style={{textAlign:"center",zIndex:1,maxWidth:680}}>
          <div style={{fontSize:11,letterSpacing:8,color:"#446",marginBottom:8,textTransform:"uppercase"}}>Reinforcement Learning Combat</div>
          <h1 style={{fontSize:52,fontWeight:900,margin:"0 0 4px",letterSpacing:-2,background:"linear-gradient(135deg,#4af 0%,#a4f 50%,#f84 100%)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",lineHeight:1}}>
            ADAPTIVE<br/>ARENA AI
          </h1>
          <div style={{fontSize:13,color:"#556",marginBottom:32,letterSpacing:2}}>V1 · ONLINE RL · EVOLVING ENEMY</div>

          <div style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.08)",borderRadius:12,padding:"20px 28px",marginBottom:28,textAlign:"left"}}>
            <div style={{fontSize:12,color:"#668",marginBottom:12,letterSpacing:2}}>CONTROLS</div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"6px 24px",fontSize:13}}>
              {[["←→ / A D","Move"],["↑ / W / Space","Jump (x2)"],["Z / J","Attack"],["X / K","Block"],["C / L","Dash"]].map(([k,v])=>(
                <div key={k} style={{display:"flex",gap:8,alignItems:"center"}}>
                  <span style={{background:"rgba(80,130,255,0.15)",border:"1px solid rgba(80,130,255,0.3)",borderRadius:4,padding:"2px 7px",fontSize:11,color:"#89b",fontFamily:"monospace",whiteSpace:"nowrap"}}>{k}</span>
                  <span style={{color:"#778",fontSize:12}}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{marginBottom:20}}>
            <div style={{fontSize:12,color:"#668",marginBottom:12,letterSpacing:2}}>SELECT AI PERSONALITY</div>
            <div style={{display:"flex",gap:8,justifyContent:"center",flexWrap:"wrap"}}>
              {PERSONALITIES.map((p,i)=>(
                <button key={p} onClick={()=>setUiState(prev=>({...prev,personality:i}))}
                  style={{padding:"8px 16px",borderRadius:6,border:`2px solid ${uiState.personality===i?PERSONALITY_COLORS[i]:"rgba(255,255,255,0.1)"}`,background:uiState.personality===i?`${PERSONALITY_COLORS[i]}22`:"transparent",color:uiState.personality===i?PERSONALITY_COLORS[i]:"#668",cursor:"pointer",fontSize:12,letterSpacing:1,transition:"all 0.15s",fontFamily:"monospace"}}>
                  {p}
                </button>
              ))}
            </div>
          </div>

          <button onClick={()=>startGame(1,uiState.personality)}
            style={{padding:"14px 48px",background:"linear-gradient(135deg,#1a3a8f,#0d1f5c)",border:"1px solid #3366ff",borderRadius:8,color:"#8af",cursor:"pointer",fontSize:16,letterSpacing:3,fontFamily:"monospace",fontWeight:"bold",boxShadow:"0 0 30px rgba(80,120,255,0.3)",transition:"all 0.2s"}}>
            ▶  ENTER ARENA
          </button>

          <div style={{marginTop:20,fontSize:11,color:"#334",letterSpacing:1}}>
            The enemy learns from every fight. The longer you play, the smarter it gets.
          </div>
        </div>
      </div>
    );
  }

  // GAME OVER SCREEN
  if(uiState.screen === "gameover") {
    const won = uiState.lastResult === "win";
    return (
      <div style={{background:"#07070f",width:"100vw",height:"100vh",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",fontFamily:"'Courier New',monospace",color:"#ccc",overflow:"hidden",position:"relative"}}>
        <div style={{position:"absolute",inset:0,background:`radial-gradient(ellipse at 50% 40%, ${won?"rgba(0,200,100,0.08)":"rgba(200,40,40,0.08)"} 0%, transparent 70%)`,pointerEvents:"none"}}/>

        <div style={{textAlign:"center",zIndex:1,maxWidth:600}}>
          <div style={{fontSize:11,letterSpacing:6,color:won?"#4a8":"#a44",marginBottom:8}}>{won?"VICTORY":"DEFEAT"}</div>
          <h2 style={{fontSize:42,fontWeight:900,margin:"0 0 24px",color:won?"#4f8":"#f44",letterSpacing:-1}}>
            {won ? "ROUND WON" : "AI WINS"}
          </h2>

          {/* Score */}
          <div style={{display:"flex",gap:24,justifyContent:"center",marginBottom:24}}>
            <div style={{textAlign:"center"}}>
              <div style={{fontSize:11,color:"#668",letterSpacing:2,marginBottom:4}}>YOU</div>
              <div style={{fontSize:36,fontWeight:900,color:"#4af"}}>{uiState.playerScore}</div>
            </div>
            <div style={{fontSize:28,color:"#334",alignSelf:"center"}}>:</div>
            <div style={{textAlign:"center"}}>
              <div style={{fontSize:11,color:"#668",letterSpacing:2,marginBottom:4}}>AI LV.{uiState.level}</div>
              <div style={{fontSize:36,fontWeight:900,color:PERSONALITY_COLORS[uiState.personality]}}>{uiState.aiScore}</div>
            </div>
          </div>

          {/* AI Stats */}
          <div style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:10,padding:"16px 24px",marginBottom:20,textAlign:"left"}}>
            <div style={{fontSize:11,color:"#668",letterSpacing:2,marginBottom:12}}>AI LEARNING STATS</div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
              {[
                ["EPISODES",uiState.episodeCount],
                ["WIN RATE",uiState.winRate+"%"],
                ["EPSILON",uiState.epsilon],
                ["LEVEL",uiState.level],
                ["PERSONALITY",PERSONALITIES[uiState.personality]],
                ["ROUNDS",uiState.playerScore+uiState.aiScore],
              ].map(([l,v])=>(
                <div key={l}>
                  <div style={{fontSize:10,color:"#446",letterSpacing:1}}>{l}</div>
                  <div style={{fontSize:14,color:"#aac",fontWeight:"bold"}}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Reward chart */}
          {rewardHistory.length > 2 && (
            <div style={{background:"rgba(0,0,0,0.3)",border:"1px solid rgba(80,80,120,0.2)",borderRadius:8,padding:"12px 16px",marginBottom:20}}>
              <div style={{fontSize:10,color:"#446",letterSpacing:2,marginBottom:8}}>REWARD HISTORY</div>
              <div style={{display:"flex",gap:2,alignItems:"flex-end",height:36}}>
                {rewardHistory.map((r,i)=>{
                  const h = Math.abs(r)/rewardMax*34+2;
                  return <div key={i} style={{flex:1,height:h,background:r>0?"rgba(80,255,130,0.6)":"rgba(255,80,80,0.6)",borderRadius:"2px 2px 0 0",transition:"height 0.3s"}}/>;
                })}
              </div>
            </div>
          )}

          <div style={{display:"flex",gap:12,justifyContent:"center"}}>
            <button onClick={continueGame}
              style={{padding:"12px 32px",background:won?"rgba(0,80,40,0.4)":"rgba(80,0,0,0.4)",border:`1px solid ${won?"#4f8":"#f44"}`,borderRadius:8,color:won?"#4f8":"#f44",cursor:"pointer",fontSize:14,letterSpacing:2,fontFamily:"monospace",fontWeight:"bold"}}>
              {won && uiState.level < 10 ? `▶ LEVEL ${uiState.level+1}` : "▶ REMATCH"}
            </button>
            <button onClick={()=>setUiState(prev=>({...prev,screen:"menu",playerScore:0,aiScore:0,level:1}))}
              style={{padding:"12px 24px",background:"transparent",border:"1px solid #334",borderRadius:8,color:"#556",cursor:"pointer",fontSize:13,letterSpacing:1,fontFamily:"monospace"}}>
              MENU
            </button>
          </div>
        </div>
      </div>
    );
  }

  // GAME SCREEN
  const timerColor = uiState.roundTimer <= 10 ? "#f44" : uiState.roundTimer <= 20 ? "#fa0" : "#aac";

  return (
    <div style={{background:"#07070f",width:"100vw",height:"100vh",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",fontFamily:"'Courier New',monospace",userSelect:"none",overflow:"hidden"}}>

      {/* TOP HUD */}
      <div style={{width:W,maxWidth:"100%",display:"flex",alignItems:"center",marginBottom:6,gap:0}}>
        {/* Player HP */}
        <div style={{flex:1}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:3,fontSize:11}}>
            <span style={{color:"#4af",letterSpacing:1}}>▶ PLAYER</span>
            <span style={{color:"#aac"}}>{uiState.playerHp}</span>
          </div>
          <div style={{height:10,background:"rgba(255,255,255,0.07)",borderRadius:5,overflow:"hidden",border:"1px solid rgba(80,160,255,0.2)"}}>
            <div style={{height:"100%",width:pct(uiState.playerHp),background:"linear-gradient(90deg,#1a6fd4,#4aafff)",borderRadius:5,transition:"width 0.1s",boxShadow:"0 0 8px rgba(80,160,255,0.6)"}}/>
          </div>
          {uiState.combo > 1 && <div style={{fontSize:10,color:"#4af",marginTop:2}}>COMBO ×{uiState.combo}</div>}
        </div>

        {/* Timer */}
        <div style={{width:80,textAlign:"center",fontSize:26,fontWeight:900,color:timerColor,letterSpacing:-1,textShadow:`0 0 10px ${timerColor}`}}>
          {uiState.roundTimer}
        </div>

        {/* AI HP */}
        <div style={{flex:1}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:3,fontSize:11}}>
            <span style={{color:"#aac"}}>{uiState.aiHp}</span>
            <span style={{color:PERSONALITY_COLORS[uiState.personality],letterSpacing:1}}>AI LV.{uiState.level} {PERSONALITIES[uiState.personality].slice(0,3).toUpperCase()} ◀</span>
          </div>
          <div style={{height:10,background:"rgba(255,255,255,0.07)",borderRadius:5,overflow:"hidden",border:`1px solid ${PERSONALITY_COLORS[uiState.personality]}33`}}>
            <div style={{height:"100%",width:pct(uiState.aiHp),background:`linear-gradient(90deg,${PERSONALITY_COLORS[uiState.personality]}99,${PERSONALITY_COLORS[uiState.personality]})`,borderRadius:5,float:"right",transition:"width 0.1s",boxShadow:`0 0 8px ${PERSONALITY_COLORS[uiState.personality]}88`}}/>
          </div>
          {uiState.aiCombo > 1 && <div style={{fontSize:10,color:PERSONALITY_COLORS[uiState.personality],marginTop:2,textAlign:"right"}}>COMBO ×{uiState.aiCombo}</div>}
        </div>
      </div>

      {/* CANVAS */}
      <div style={{position:"relative",boxShadow:"0 0 60px rgba(40,60,200,0.3)",border:"1px solid rgba(80,100,200,0.2)",borderRadius:4}}>
        <canvas ref={canvasRef} width={W} height={H} style={{display:"block"}}/>

        {/* Overlay: AI action badge */}
        <div style={{position:"absolute",top:8,right:8,background:"rgba(0,0,0,0.6)",border:`1px solid ${PERSONALITY_COLORS[uiState.personality]}44`,borderRadius:4,padding:"3px 8px",fontSize:10,color:PERSONALITY_COLORS[uiState.personality],letterSpacing:1}}>
          AI: {uiState.aiAction.toUpperCase()}
        </div>
      </div>

      {/* BOTTOM STATS */}
      <div style={{width:W,maxWidth:"100%",display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginTop:6,gap:12}}>
        {/* Score */}
        <div style={{display:"flex",gap:16,alignItems:"center"}}>
          <div style={{fontSize:11,color:"#4af",fontWeight:"bold"}}>{uiState.playerScore}</div>
          <div style={{fontSize:10,color:"#334"}}>vs</div>
          <div style={{fontSize:11,color:PERSONALITY_COLORS[uiState.personality],fontWeight:"bold"}}>{uiState.aiScore}</div>
        </div>

        {/* RL Stats mini */}
        <div style={{display:"flex",gap:12,fontSize:10,color:"#446"}}>
          <span>ε={uiState.epsilon}</span>
          <span style={{color:uiState.winRate>50?"#f44":"#4a8"}}>AI {uiState.winRate}%</span>
          <span>ep={uiState.episodeCount}</span>
        </div>

        {/* Reward sparkline */}
        {rewardHistory.length > 2 && (
          <div style={{display:"flex",gap:1,alignItems:"flex-end",height:20}}>
            {rewardHistory.slice(-12).map((r,i)=>{
              const h = Math.abs(r)/rewardMax*18+2;
              return <div key={i} style={{width:4,height:h,background:r>0?"rgba(80,255,130,0.5)":"rgba(255,80,80,0.5)",borderRadius:"1px 1px 0 0"}}/>;
            })}
          </div>
        )}
      </div>

      <div style={{marginTop:4,fontSize:10,color:"#334",letterSpacing:1}}>
        Z·ATTACK &nbsp;·&nbsp; X·BLOCK &nbsp;·&nbsp; C·DASH &nbsp;·&nbsp; ↑·JUMP
      </div>
    </div>
  );
}
