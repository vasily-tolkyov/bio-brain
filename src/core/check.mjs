/** 验证脚本的断言与汇总：每条检查打印 ✓/✗ 与实测数字，done() 返回是否全部通过 */
export function checker(title) {
  const rows = [];
  console.log(`\n=== ${title} ===`);
  return {
    check(name, ok, detail = '') {
      rows.push({ name, ok: !!ok });
      console.log(`${ok ? '✓' : '✗'} ${name}${detail ? `（${detail}）` : ''}`);
      return !!ok;
    },
    note(text) { console.log(`  · ${text}`); },
    done() {
      const failed = rows.filter((r) => !r.ok);
      console.log(failed.length ? `✗ ${title}：${failed.length}/${rows.length} 项未通过` : `✓ ${title}：${rows.length} 项全部通过`);
      return failed.length === 0;
    },
  };
}

export const fmt = (x, d = 2) => (Number.isFinite(x) ? (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(d) : x.toFixed(d)) : String(x));
