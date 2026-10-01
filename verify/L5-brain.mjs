/** 第 5 层验证：方向环（maleCNS 核）、海马、前额叶工作记忆、皮层规则学习，以及各模块工作状态的 ATP */
import { HeadDirectionRing, pfl3Turn, wrapPi, KERNEL_DATA, oddPart, evenPart } from '../src/L5-brain/head-direction.mjs';
import { Hippocampus, overlapFrac } from '../src/L5-brain/hippocampus.mjs';
import { PrefrontalWM } from '../src/L5-brain/prefrontal.mjs';
import { PredictionErrorLearner, SparseExpansion } from '../src/L5-brain/cortex.mjs';
import { rateStateAtpPerSecond } from '../src/L4-network/metabolism.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { checker, fmt } from '../src/core/check.mjs';

const DEG = 180 / Math.PI;
const pearson = (x, y) => { const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n; let a = 0, b = 0, c = 0; for (let i = 0; i < n; i++) { a += (x[i] - mx) * (y[i] - my); b += (x[i] - mx) ** 2; c += (y[i] - my) ** 2; } return a / Math.sqrt(b * c); };

function verifyHeadDirection(c, costs) {
  const run = (ring, n, opts = {}) => { for (let k = 0; k < n; k++) ring.step(0.1, opts); };
  const ring = new HeadDirectionRing();
  run(ring, 200, { landmark: 1.0 });
  run(ring, 1000);
  const p1 = ring.bump();
  run(ring, 2000);
  const p2 = ring.bump();
  const drift = Math.abs(wrapPi(p2.pos - p1.pos)) * DEG;
  c.check('方向环（核 = maleCNS 实测的 EPG→EPG 有效作用的偶部）：撤掉地标后凸包留在原处——连续吸引子', p2.strength > 0.6 && drift < 3,
    `强度 ${fmt(p2.strength)}，200τ 内漂移 ${fmt(drift, 1)}°，约 ${fmt(p2.active, 0)}/32 个单元活跃`);

  const raw = new HeadDirectionRing({ symmetric: false });
  run(raw, 200, { landmark: 1.0 });
  run(raw, 1000);
  const q1 = raw.bump();
  run(raw, 2000);
  const rawDrift = Math.abs(wrapPi(raw.bump().pos - q1.pos)) * DEG;
  const odd = oddPart(KERNEL_DATA.direct), even = evenPart(KERNEL_DATA.direct);
  c.check('带上实测的左右不对称（奇部）后，黑暗里凸包被一股恒定的"风"吹着走——真实果蝇要靠视觉地标不断重校准', rawDrift > 5 * Math.max(drift, 1),
    `奇部在 ±45° 处 ${fmt(odd[KERNEL_DATA.binsDeg.indexOf(45)], 3)}（偶部 ${fmt(even[KERNEL_DATA.binsDeg.indexOf(45)], 2)}），200τ 漂移 ${fmt(rawDrift, 1)}°`);

  const oms = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], vel = [];
  for (const om of oms) {
    const rr = new HeadDirectionRing();
    run(rr, 300, { landmark: 0 });
    let prev = rr.bump().pos, acc = 0;
    for (let k = 0; k < 500; k++) { rr.step(0.1, { omega: om }); const p = rr.bump().pos; acc += wrapPi(p - prev); prev = p; }
    vel.push((acc * DEG) / 50);
  }
  const slope = vel.reduce((s, v, i) => s + v * oms[i], 0) / oms.reduce((s, o) => s + o * o, 0);
  c.check('PEN 左右通道被角速度调制：凸包转速与角速度成正比（有向通道）', pearson(oms, vel) > 0.99, `转速 ${vel.map((v) => fmt(v)).join(' / ')} °/τ，斜率 ${fmt(slope)} °/τ`);

  const pi = new HeadDirectionRing();
  run(pi, 300, { landmark: 0 });
  let truth = pi.bump().pos, prev = truth, est = truth;
  for (let k = 0; k < 4000; k++) {
    const t = k * 0.1, om = 0.12 + 0.12 * Math.sin(t / 40) + 0.06 * Math.sin(t / 13 + 1);
    pi.step(0.1, { omega: om });
    truth += (slope * om * 0.1) / DEG;
    const p = pi.bump().pos; est += wrapPi(p - prev); prev = p;
  }
  const piErr = Math.abs(wrapPi(est - truth)) * DEG;
  c.check('路径积分：黑暗中按变化的角速度转了 400τ（累计 > 100°），凸包与真实朝向的误差 < 25°', piErr < 25 && Math.abs(truth) * DEG > 100, `累计转过 ${fmt(Math.abs(truth) * DEG, 0)}°，误差 ${fmt(piErr, 1)}°`);
  run(pi, 300, { landmark: 2.0 });
  c.check('地标重校准：看到地标后凸包被拉到地标方向', Math.abs(wrapPi(pi.bump().pos - 2.0)) * DEG < 10, `偏差 ${fmt(Math.abs(wrapPi(pi.bump().pos - 2.0)) * DEG, 1)}°`);

  const t0 = pfl3Turn(0, 0), tl = pfl3Turn(-0.5, 0), tr = pfl3Turn(0.5, 0), tb = pfl3Turn(Math.PI, 0), tb1 = pfl3Turn(Math.PI - 0.3, 0);
  c.check('PFL3 转向读出：朝向目标时为零；偏一侧就往回转（稳定）；背对目标也为零但一推就走（不稳定）', Math.abs(t0) < 1e-9 && tl > 0 && tr < 0 && Math.abs(tb) < 1e-9 && tb1 < 0,
    `偏 −29°：${fmt(tl)}，偏 +29°：${fmt(tr)}，背对：${fmt(tb, 3)}，背对偏 −17°：${fmt(tb1)}`);
  const atpBump = rateStateAtpPerSecond(ring.r, costs), atpSilent = rateStateAtpPerSecond(new Float64Array(32), costs);
  c.note(`方向环：凸包状态 ${fmt(atpBump)} ATP/s，全静默 ${fmt(atpSilent)} ATP/s`);
}

function verifyHippocampus(c, costs) {
  const rng = makeRng(5);
  const pattern = (n, k) => { const x = new Float64Array(n); for (const i of rng.shuffle([...Array(n).keys()]).slice(0, k)) x[i] = 1; return x; };
  const hc = new Hippocampus();
  const pairs = Array.from({ length: 10 }, () => {
    const base = pattern(100, 20), v = base.slice();
    const on = [...base.keys()].filter((i) => base[i]), off = rng.shuffle([...base.keys()].filter((i) => !base[i]));
    for (let k = 0; k < 4; k++) { v[on[k]] = 0; v[off[k]] = 1; }
    return [overlapFrac(base, v), overlapFrac(hc.dg(base), hc.dg(v))];
  });
  const ecOv = pairs.reduce((s, p) => s + p[0], 0) / pairs.length, dgOv = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
  c.check('海马 DG：模式分离——80% 重叠的两个输入，在齿状回里只剩很少的共同细胞', dgOv < ecOv - 0.3, `EC 重叠 ${fmt(ecOv)} → DG 重叠 ${fmt(dgOv)}`);

  const stored = Array.from({ length: 10 }, () => pattern(100, 20));
  stored.forEach((p) => hc.encode(p));
  const rec = stored.map((p, m) => { const on = [...p.keys()].filter((i) => p[i]); const cue = new Float64Array(100); for (const i of on.slice(0, 10)) cue[i] = 1; const r = hc.recall(cue); return { ca3: overlapFrac(r.ca3, hc.episodes[m].ca3), ec: overlapFrac(r.ecOut, p) }; });
  c.check('海马 CA3：每条经历只见一次就存下；给一半线索，CA3 补全并经 CA1 还原出完整输入', rec.every((r) => r.ca3 > 0.9 && r.ec > 0.9), `CA3 ${fmt(Math.min(...rec.map((r) => r.ca3)))}，还原 EC ${fmt(Math.min(...rec.map((r) => r.ec)))}（最差）`);
  const fam = stored.map((p) => hc.novelty(p)), nov = Array.from({ length: 10 }, () => hc.novelty(pattern(100, 20)));
  c.check('海马 CA1：熟悉的输入"提取 = 实际"，新输入两者对不上（新奇信号）', Math.max(...fam) < 0.2 && Math.min(...nov) > 0.5, `熟悉 ≤ ${fmt(Math.max(...fam))}，新 ≥ ${fmt(Math.min(...nov))}`);

  const hs = new Hippocampus({ seed: 2 });
  const codes = hs.encodeSequence(Array.from({ length: 6 }, () => pattern(100, 20)));
  const rp = hs.replay({ start: codes[0].ca3 }), rpNoSeq = hs.replay({ start: codes[0].ca3, seqGain: 0 }), rpNoF = hs.replay({ start: codes[0].ca3, fatigue: 0 });
  c.check('海马重放：休息时按经历的顺序重放（CA3 里的不对称连接 + 疲劳 = 有向通道）；去掉疲劳走一步就卡住，去掉不对称连接根本不动', rp.join(',') === '0,1,2,3,4,5' && rpNoSeq.length === 1 && rpNoF.length <= 2,
    `重放 ${rp.join('→')}；无序列连接 ${rpNoSeq.join('→')}；无疲劳 ${rpNoF.join('→')}`);
  const dgCost = rateStateAtpPerSecond(hc.episodes[0].dg, costs) / hc.nDG, ca3Cost = rateStateAtpPerSecond(hc.episodes[0].ca3, costs) / hc.nCA3;
  c.note(`稀疏码的代谢：DG 活跃 ${fmt(hc.kDG / hc.nDG * 100, 0)}%、CA3 ${fmt(hc.kCA3 / hc.nCA3 * 100, 0)}%；每个细胞平均 ${fmt(dgCost)} / ${fmt(ca3Cost)} ATP/s`);
}

function verifyPrefrontal(c, costs) {
  const wm = new PrefrontalWM();
  const run = (n, opts = {}) => { for (let k = 0; k < n; k++) wm.step(0.1, opts); };
  const idle = rateStateAtpPerSecond(Float64Array.from(wm.activity()), costs);
  run(100, { gates: [1, 0, 0], inputs: [3, null, null] });
  run(2000);
  const held = wm.held(0);
  const holding = rateStateAtpPerSecond(Float64Array.from(wm.activity()), costs);
  run(100, { gates: [0, 0, 0], inputs: [5, null, null] }); run(200);
  const afterDistractor = wm.held(0);
  run(100, { gates: [1, 0, 0], inputs: [5, null, null] }); run(200);
  const updated = wm.held(0);
  run(100, { gates: [0, 1, 0], inputs: [null, 2, null] }); run(500);
  c.check('前额叶：门开时载入条目，门关后靠吸引子维持 200τ', held === 3, `记住 ${held}`);
  c.check('门关时的干扰输入进不来；门开时才能更新', afterDistractor === 3 && updated === 5, `干扰后 ${afterDistractor}，门开更新后 ${updated}`);
  c.check('各槽位相互独立', wm.held(0) === 5 && wm.held(1) === 2 && wm.held(2) === null, `槽位 ${[0, 1, 2].map((s) => wm.held(s) ?? '空').join(' / ')}`);
  c.check('记住东西要付持续的代价（工作记忆比空闲贵）', holding > idle, `空闲 ${fmt(idle)}，记住一个条目 ${fmt(holding)} ATP/s`);
}

function verifyCortex(c) {
  const rng = makeRng(9);
  const rw = new PredictionErrorLearner(2, 1, { alpha: 0.05 });
  let co = 0, nB = 0;
  for (let t = 0; t < 2000; t++) {
    const both = rng.uniform() < 0.5, x = both ? [1, 1] : [0, 1], o = both ? 1 : 0;
    rw.learn(x, o);
    nB++; co += o;
  }
  c.check('皮层规则（R2 的差分）：预测误差学习只把真正有用的线索学进去——B 总伴随出现、但单独出现时没有结果，被剔除', rw.W[0][0] > 0.9 && rw.W[0][1] < 0.1,
    `w_A ${fmt(rw.W[0][0])}，w_B ${fmt(rw.W[0][1])}（单纯共现统计会给 B ${fmt(co / nB)}）`);
  const bl = new PredictionErrorLearner(2, 1, { alpha: 0.05 }), ctl = new PredictionErrorLearner(2, 1, { alpha: 0.05 });
  for (let t = 0; t < 300; t++) bl.learn([1, 0], 1);
  for (let t = 0; t < 300; t++) { bl.learn([1, 1], 1); ctl.learn([1, 1], 1); }
  c.check('阻断：结果已被旧线索预测，新加入的线索学不到（对照组能学到）', bl.W[0][1] < 0.1 && ctl.W[0][1] > 0.3, `阻断组 w_X ${fmt(bl.W[0][1])}，对照组 ${fmt(ctl.W[0][1])}`);

  const trials = [[1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [0, 1, 0, 1, 1]];
  const lin = new PredictionErrorLearner(4, 1, { alpha: 0.05 }), exp = new SparseExpansion(4, 200, { fan: 2, k: 0.05, seed: 3 }), lin2 = new PredictionErrorLearner(200, 1, { alpha: 0.2 });
  for (let t = 0; t < 4000; t++) { const tr = trials[t % 4]; lin.learn(tr.slice(0, 4), tr[4]); lin2.learn(exp.encode(tr.slice(0, 4)), tr[4]); }
  const pl = trials.map((tr) => lin.predict(tr.slice(0, 4))[0]), pe = trials.map((tr) => lin2.predict(exp.encode(tr.slice(0, 4)))[0]);
  const errL = Math.max(...pl.map((p, i) => Math.abs(p - trials[i][4]))), errE = Math.max(...pe.map((p, i) => Math.abs(p - trials[i][4])));
  c.check('合取规则（"状态 s₁ 且动作 a₁ 才有结果"）：线性学不准，经稀疏随机扩张后同一规则学得准', errL > 0.2 && errE < 0.05, `线性 ${pl.map((p) => fmt(p)).join('/')}，扩张后 ${pe.map((p) => fmt(p)).join('/')}`);

  const pattern = (n, k) => { const x = new Float64Array(n); for (const i of rng.shuffle([...Array(n).keys()]).slice(0, k)) x[i] = 1; return x; };
  const S1 = Array.from({ length: 8 }, () => pattern(40, 10)), S2 = Array.from({ length: 8 }, () => pattern(40, 10));
  const T1 = S1.map(() => (rng.uniform() < 0.5 ? 1 : 0)), T2 = S2.map(() => (rng.uniform() < 0.5 ? 1 : 0));
  const mse = (L, S, T) => S.reduce((s, x, i) => s + (L.predict(x)[0] - T[i]) ** 2, 0) / S.length;
  const blocked = new PredictionErrorLearner(40, 1, { alpha: 0.1 }), inter = new PredictionErrorLearner(40, 1, { alpha: 0.1 });
  for (let e = 0; e < 300; e++) S1.forEach((x, i) => { blocked.learn(x, T1[i]); inter.learn(x, T1[i]); });
  for (let e = 0; e < 300; e++) { S2.forEach((x, i) => { blocked.learn(x, T2[i]); inter.learn(x, T2[i]); }); S1.forEach((x, i) => inter.learn(x, T1[i])); }
  const mB = mse(blocked, S1, T1), mI = mse(inter, S1, T1);
  c.check('互补学习系统：只学新任务会冲掉旧规则；海马把旧经历交错回放给皮层，新旧都保住', mB > 10 * Math.max(mI, 1e-4) && mse(inter, S2, T2) < 0.01,
    `旧任务误差：按块学 ${fmt(mB, 3)}，交错回放 ${fmt(mI, 4)}；新任务 ${fmt(mse(inter, S2, T2), 4)}`);
}

export function verifyL5() {
  const c = checker('第 5 层 脑结构');
  const costs = mammalianCosts();
  verifyHeadDirection(c, costs);
  verifyHippocampus(c, costs);
  verifyPrefrontal(c, costs);
  verifyCortex(c);
  return { ok: c.done() };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL5().ok ? 0 : 1);
