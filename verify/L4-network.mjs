/** 第 4 层验证：网络的能量地形（对称部分）、吸引子、代谢账，以及有向通道（反对称部分 + 疲劳）。
 * 最后一项回到脉冲神经元：兴奋/抑制集合的双稳态及其 ATP。 */
import { RateNetwork, overlap } from '../src/L4-network/rate-network.mjs';
import { randomPatterns, hebbianWeights, boundedHebbianWeights } from '../src/L4-network/learning.mjs';
import { rateStateAtpPerSecond } from '../src/L4-network/metabolism.mjs';
import { SpikingCircuit } from '../src/L3-plasticity/circuit.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { checker, fmt } from '../src/core/check.mjs';

const settle = (net, steps, I = null) => { for (let k = 0; k < steps; k++) net.step(0.1, I); };
const active = (r) => r.reduce((s, v) => s + v, 0);

function cueRecall(net, pats, f, m, { frac = 0.4, wrong = 0, rng }) {
  net.r.fill(0); net.a.fill(0);
  const on = [...Array(net.N).keys()].filter((i) => pats[m][i]), off = [...Array(net.N).keys()].filter((i) => !pats[m][i]);
  for (const i of on.slice(0, Math.max(1, Math.round(frac * on.length)))) net.r[i] = 1;
  for (const i of rng.shuffle(off).slice(0, wrong)) net.r[i] = 1;
  const E = [];
  for (let k = 0; k < 300; k++) { net.step(0.1); E.push(net.energy()); }
  return { ov: overlap(net.r, pats[m], f), E };
}

/** 有向通道：顺序访问的模式（重叠 > 0.6 记为进入）、每次停留的进入/离开时能量 */
function runSequence(pats, f, { lambda, kA, bias = -0.25, noise = 0.02, steps = 6000, cyclic = false, seed = 3 }) {
  const rng = makeRng(seed), N = pats[0].length;
  const s = new RateNetwork(N, { bias, kA, tauA: 30 });
  s.setWeights(hebbianWeights(pats, f, { lambda, cyclic }));
  s.r.set(pats[0]);
  const visits = [], dwells = [];
  let cur = -1, eIn = null, eLast = null;
  for (let k = 0; k < steps; k++) {
    s.step(0.1, null, { noise, rng });
    const ov = pats.map((p) => overlap(s.r, p, f));
    const b = ov.indexOf(Math.max(...ov));
    if (ov[b] > 0.6 && visits.at(-1) !== b) {
      if (eIn !== null && eLast !== null) dwells.push({ eIn, eLast });
      visits.push(b); cur = b; eIn = null; eLast = null;
    }
    if (b === cur && ov[b] > 0.9) { const E = s.energy(null, pats[cur]); if (eIn === null) eIn = E; eLast = E; }
  }
  return { visits, dwells, asym: s.asymmetry() };
}
const currents = (v, P) => { let fw = 0, bw = 0; for (let i = 1; i < v.length; i++) { const d = (v[i] - v[i - 1] + P) % P; if (d === 1) fw++; else if (d === P - 1) bw++; } return { fw, bw, n: v.length - 1 }; };

export function verifyL4() {
  const c = checker('第 4 层 网络：吸引子与有向通道');
  const rng = makeRng(7);
  const costs = mammalianCosts();

  // ── 对称部分：能量地形与吸引子 ──
  const N = 200, f = 0.1, P = 5;
  const pats = randomPatterns(P, N, f, rng);
  const net = new RateNetwork(N, { bias: -0.25 });
  net.setWeights(hebbianWeights(pats, f));
  c.check('协方差赫布规则得到的权重是对称的（反对称占比 0）', net.asymmetry() < 1e-12, `反对称占比 ${fmt(net.asymmetry(), 3)}`);
  const rec = pats.map((_, m) => cueRecall(net, pats, f, m, { rng }));
  c.check('补全：只给 40% 的模式，网络自己落到完整模式（吸引子）', rec.every((x) => x.ov > 0.95), `重叠 ${rec.map((x) => fmt(x.ov, 3)).join(' / ')}`);
  const maxRise = Math.max(...rec.map((x) => Math.max(...x.E.slice(1).map((e, i) => e - x.E[i]))));
  c.check('李雅普诺夫性：对称权重下能量沿动力学只降不升', maxRise < 1e-9, `最大单步上升 ${fmt(maxRise, 2)}；能量 ${fmt(rec[0].E[0])} → ${fmt(rec[0].E.at(-1))}`);

  const census = {};
  for (let k = 0; k < 30; k++) {
    net.a.fill(0);
    for (let i = 0; i < N; i++) net.r[i] = rng.uniform() < 0.15 ? 1 : 0;
    settle(net, 400);
    const ov = pats.map((p) => overlap(net.r, p, f)), b = ov.indexOf(Math.max(...ov));
    const key = active(net.r) < 3 ? '静默' : ov[b] > 0.9 ? `模式${b}` : '混合';
    census[key] = (census[key] ?? 0) + 1;
  }
  c.check('从 30 个随机起点出发，网络都落到存储的模式或静默（没有伪吸引子）', !census['混合'], Object.entries(census).map(([k, v]) => `${k}×${v}`).join(' '));

  const sameSize = Array.from({ length: 20 }, () => { const x = new Float64Array(N); for (const i of rng.shuffle([...Array(N).keys()]).slice(0, f * N)) x[i] = 1; return net.energy(null, x); });
  const ePat = pats.map((p) => net.energy(null, p));
  c.check('存储的模式是能量低谷：比同样大小的随机状态能量低', Math.max(...ePat) < Math.min(...sameSize), `模式 ${fmt(Math.max(...ePat))}，随机 ≥ ${fmt(Math.min(...sameSize))}`);

  const atpSilent = rateStateAtpPerSecond(new Float64Array(N), costs), atpPat = rateStateAtpPerSecond(pats[0], costs);
  c.check('代谢账：吸引子状态比静默贵（吸引子不是代谢最低点，静默才是）', atpPat > 2 * atpSilent, `静默 ${fmt(atpSilent)}，模式 ${fmt(atpPat)} ATP/s`);

  // 排除静默后：能完成补全的最稀疏编码就是"能工作的最省能状态"（单个突触强度有上限，小集合撑不起来）
  const scan = [0.02, 0.05, 0.1, 0.2].map((ff) => {
    const r2 = makeRng(21);
    const ps = randomPatterns(P, N, ff, r2);
    const nn = new RateNetwork(N, { bias: -0.25 });
    nn.setWeights(boundedHebbianWeights(ps, { w: 0.05, J0: 1 }));
    const ok = ps.every((_, m) => cueRecall(nn, ps, ff, m, { frac: 0.4, wrong: 2, rng: r2 }).ov > 0.9);
    return { f: ff, ok, atp: rateStateAtpPerSecond(ps[0], costs) };
  });
  const working = scan.filter((s) => s.ok), cheapest = working.sort((a, b) => a.atp - b.atp)[0];
  c.check('排除静默这个平凡解：存在一个能工作的最省能吸引子——能完成补全的最稀疏编码；更稀疏就补全不了', !scan[0].ok && cheapest && cheapest.f < 0.2,
    scan.map((s) => `f=${s.f} ${s.ok ? '能补全' : '补全失败'} ${fmt(s.atp)} ATP/s`).join('；'));

  // ── 反对称部分 + 疲劳：有向通道 ──
  const sp = randomPatterns(6, 100, 0.1, makeRng(11));
  const both = runSequence(sp, 0.1, { lambda: 0.3, kA: 0.5 });
  const asymOnly = runSequence(sp, 0.1, { lambda: 0.3, kA: 0 });
  const fatigueOnly = runSequence(sp, 0.1, { lambda: 0, kA: 0.5 });
  c.check('不对称连接（模式 μ → μ+1）+ 疲劳：网络按学过的顺序依次走过每个吸引子', both.visits.join(',') === '0,1,2,3,4,5', `访问 ${both.visits.join('→')}，W 的反对称占比 ${fmt(both.asym, 2)}`);
  c.check('只有不对称、没有疲劳：走不下去（当前的井太深，风吹不动）', asymOnly.visits.length <= 2, `访问 ${asymOnly.visits.join('→')}`);
  c.check('只有疲劳、没有不对称：井变浅后没有方向，不会沿序列前进', fatigueOnly.visits.length <= 1, `访问 ${fatigueOnly.visits.join('→') || '—'}`);
  const rising = both.dwells.filter((d) => d.eLast > d.eIn).length;
  c.check('能量图景：停在一个井里时，疲劳把井底慢慢抬高（当前模式在此刻适应下的能量，离开前高于刚进入时）', both.dwells.length >= 4 && rising >= 0.8 * both.dwells.length,
    `${rising}/${both.dwells.length} 次停留能量上升，例：${both.dwells.slice(0, 3).map((d) => `${fmt(d.eIn)}→${fmt(d.eLast)}`).join('，')}`);

  const sym = currents(runSequence(sp, 0.1, { lambda: 0, kA: 0.5, bias: -0.15, noise: 0.15, steps: 30000, cyclic: true }).visits, 6);
  const dir = currents(runSequence(sp, 0.1, { lambda: 0.3, kA: 0.5, bias: -0.15, noise: 0.15, steps: 30000, cyclic: true }).visits, 6);
  const Jdir = (dir.fw - dir.bw) / (dir.fw + dir.bw || 1);
  c.check('细致平衡：对称网络在噪声下井间乱跳，正向与反向一样多', sym.n > 50 && Math.abs(sym.fw - sym.bw) <= Math.max(4, 0.4 * (sym.fw + sym.bw)), `${sym.n} 次转移，正向 ${sym.fw}、反向 ${sym.bw}`);
  c.check('有向通道打破细致平衡：几乎只沿学过的方向转移（净环流接近 1）', Jdir > 0.9 && dir.fw > 100, `${dir.n} 次转移，正向 ${dir.fw}、反向 ${dir.bw}，净环流 ${fmt(Jdir, 2)}`);

  // ── 回到脉冲神经元：兴奋/抑制集合的双稳态 ──
  const NE = 40, NI = 10, synapses = [];
  for (let i = 0; i < NE; i++) for (let j = 0; j < NE; j++) if (i !== j) synapses.push({ pre: i, post: j, contacts: 10, pRelease: 0.2, tauRec: 100, scale: 3 });
  for (let i = 0; i < NE; i++) for (let k = 0; k < NI; k++) synapses.push({ pre: i, post: NE + k, contacts: 10, pRelease: 0.2, tauRec: 100, scale: 3 });
  for (let k = 0; k < NI; k++) for (let j = 0; j < NE; j++) synapses.push({ pre: NE + k, post: j, kind: 'inh', contacts: 10, pRelease: 0.5, tauRec: 300, scale: 1 });
  const sc = new SpikingCircuit({ costs, seed: 4, neurons: Array.from({ length: NE + NI }, (_, i) => ({ p: i < NE ? {} : { C: 100, tRef: 1 } })), synapses });
  const meanE = (t0, t1) => sc.rates(t0, t1).slice(0, NE).reduce((s, v) => s + v, 0) / NE;
  sc.run(100); sc.ledger.reset(); let t0 = sc.t; sc.run(500); const silentAtp = sc.ledger.total() / 0.5, rSilent = meanE(t0, sc.t);
  sc.run(50, { extra: () => Array.from({ length: NE + NI }, (_, i) => (i < NE ? 250 : 0)) });
  sc.ledger.reset(); t0 = sc.t; sc.run(1000); const upAtp = sc.ledger.total(), rUp = meanE(t0 + 300, sc.t);
  sc.run(50, { extra: () => Array.from({ length: NE + NI }, (_, i) => (i < NE ? -600 : 0)) });
  t0 = sc.t; sc.run(500); const rOff = meanE(t0 + 200, sc.t);
  c.check('脉冲神经元集合是双稳态：短暂刺激后自持放电，短暂抑制后回到静默（两个吸引子）', rSilent === 0 && rUp > 20 && rOff === 0, `静默 ${fmt(rSilent, 0)} Hz → 刺激后 ${fmt(rUp, 0)} Hz → 抑制后 ${fmt(rOff, 0)} Hz`);
  c.check('自持的吸引子状态要持续付 ATP（远高于静默）', upAtp > 5 * silentAtp, `${fmt(upAtp)} vs ${fmt(silentAtp)} ATP/s`);
  c.note('已知缺口：这个最简单的兴奋/抑制集合只有高频（约 100 Hz）持续态；皮层式的低频持续活动需要 NMDA 饱和或短时程易化等机制');
  return { ok: c.done() };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4().ok ? 0 : 1);
