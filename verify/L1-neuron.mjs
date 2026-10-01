/** 第 1 层验证：HH 神经元的静息、阈值、放电、ATP 账；点神经元的 f–I 与适应 */
import { HHNeuron, HH_SQUID, calibrateHH, costsForArea } from '../src/L1-neuron/hh.mjs';
import { LIFNeuron, rateAt, LIF_DEFAULT } from '../src/L1-neuron/lif.mjs';
import { EnergyLedger } from '../src/core/ledger.mjs';
import { checker, fmt } from '../src/core/check.mjs';

export function verifyL1() {
  const c = checker('第 1 层 单个神经元');
  const dt = 0.01;

  const cal = calibrateHH();
  c.check('静息电位约 −65 mV', Math.abs(cal.vRest + 65) < 1, `${fmt(cal.vRest)} mV`);
  c.check('1 ms 强脉冲引发一次动作电位，峰值 > +20 mV', cal.spikes === 1 && cal.vPeak > 20, `放电 ${cal.spikes} 次，峰值 ${fmt(cal.vPeak)} mV`);

  const run = (I, ms = 500) => { const n = new HHNeuron(); for (let t = 0; t < 100; t += dt) n.step(dt, 0); let k = 0; for (let t = 0; t < ms; t += dt) if (n.step(dt, I)) k++; return (k * 1000) / ms; };
  const f2 = run(2), f10 = run(10), f20 = run(20);
  c.check('阈下电流（2 µA/cm²）不持续放电', f2 <= 2, `${fmt(f2, 1)} Hz`);
  c.check('10 µA/cm² 持续放电 40–100 Hz，电流越大越快', f10 >= 40 && f10 <= 100 && f20 > f10, `${fmt(f10, 1)} Hz → ${fmt(f20, 1)} Hz`);

  c.check('6.3 °C：一次放电的 Na⁺ 内流约为把膜电容充到峰值所需最少电荷的 11 倍（文献 1098 nC/cm²、11.2 倍）', cal.overlap > 9 && cal.overlap < 14,
    `Na⁺ 电荷 ${fmt(cal.spikeNaNanoC, 0)} nC/cm²，最少 ${fmt(cal.minimalNaNanoC, 0)} nC/cm²，重叠系数 ${fmt(cal.overlap)}`);
  const warm = calibrateHH({ T: 18.5 });
  c.check('18.5 °C：通道变快、Na⁺/K⁺ 电流重叠变少，重叠系数降到约 4（文献 18 °C 3.85 倍）', warm.spikes === 1 && warm.overlap > 2.5 && warm.overlap < 5.5 && warm.spikeNaNanoC < cal.spikeNaNanoC / 2,
    `Na⁺ 电荷 ${fmt(warm.spikeNaNanoC, 0)} nC/cm²，重叠系数 ${fmt(warm.overlap)}`);
  c.note(`每 cm² 膜：一次放电 ${fmt(cal.spikeAtpPerCm2)} ATP，静息每秒 ${fmt(cal.restAtpPerMsCm2 * 1000)} ATP`);

  const area = 20000; // µm²：一个中等大小神经元（胞体 + 树突）
  const costs = costsForArea(cal, area);
  const restPerS = costs.atpRestPerMs * 1000;
  const breakEven = restPerS / costs.atpPerSpike;
  c.note(`膜面积 ${area} µm² 的神经元：一次放电 ${fmt(costs.atpPerSpike)} ATP，静息每秒 ${fmt(restPerS)} ATP`);
  c.check('静息成本不可忽略：放电频率要到若干 Hz 才与静息成本相当', breakEven > 0.5 && breakEven < 200, `静息成本 = ${fmt(breakEven, 1)} 次放电/秒的成本`);

  const ledger = new EnergyLedger();
  const n = new LIFNeuron({}, costs, ledger);
  let k = 0;
  for (let t = 0; t < 1000; t += 0.05) if (n.step(0.05, 400)) k++;
  const expect = costs.atpRestPerMs * n.t + k * costs.atpPerSpike;
  c.check('点神经元账本 = 静息 × 时长 + 放电次数 × 单次成本', Math.abs(ledger.total() - expect) / expect < 1e-6,
    `${k} 次放电，静息 ${fmt(ledger.shares().rest * 100, 0)}% / 放电 ${fmt(ledger.shares().spike * 100, 0)}%`);

  const fi = [100, 150, 200, 300, 400, 600].map((I) => rateAt(LIF_DEFAULT, I));
  const rheo = LIF_DEFAULT.gL * (LIF_DEFAULT.Vth - LIF_DEFAULT.EL);
  c.check('f–I 曲线：阈下（< g_L·(V_th−E_L)）不放电，之上单调上升', fi[0] === 0 && fi.slice(2).every((v, i, a) => i === 0 || v > a[i - 1]),
    `阈值电流 ${rheo} pA；${fi.map((v) => fmt(v, 0)).join(' / ')} Hz`);

  const ad = new LIFNeuron({ a: 2, b: 60, tauW: 200 });
  const sp = [];
  for (let t = 0; t < 1000; t += 0.05) if (ad.step(0.05, 500)) sp.push(t);
  const isi = sp.slice(1).map((v, i) => v - sp[i]);
  c.check('适应电流让持续放电越来越慢（"累"）', isi.length > 3 && isi.at(-1) / isi[0] > 1.5, `首个间隔 ${fmt(isi[0], 1)} ms → 末个 ${fmt(isi.at(-1), 1)} ms`);
  return { ok: c.done(), cal, costs };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL1().ok ? 0 : 1);
