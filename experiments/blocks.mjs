import fs from 'fs';
import { buildNetwork, CFG } from '../demo/fig2-network.mjs';

const { EE } = buildNetwork(CFG);
const s = JSON.parse(fs.readFileSync('cache/trained-seed1.json', 'utf8'));
const g = (i) => (i < 80 ? 'ABCD'[Math.floor(i / 20)] : 'X');
const acc = {};
for (let k = 0; k < EE.pre.length; k++) { const key = `${g(EE.pre[k])}→${g(EE.post[k])}`; (acc[key] ??= []).push(s.rho[k]); }
const m = (x) => (x.reduce((a, b) => a + b, 0) / x.length).toFixed(2);
for (const a of 'ABCDX') console.log([...'ABCDX'].map((b) => `${a}→${b} ${m(acc[`${a}→${b}`])}`).join('  '));
