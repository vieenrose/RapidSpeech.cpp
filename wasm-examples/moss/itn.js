/* Conservative inverse text normalization for zh-TW transcripts (browser).
 * Mirrors src/distil_vibevoice/data/itn.py: spoken numbers -> digits,
 * percentages, guarding idioms and lone ordinal chars. Self-contained
 * zh-number parser (no cn2an dependency). Written form only; deterministic. */

const DIGIT = { 零: 0, 〇: 0, 一: 1, 二: 2, 兩: 2, 三: 3, 四: 4, 五: 5,
                六: 6, 七: 7, 八: 8, 九: 9 };
const UNIT = { 十: 10, 百: 100, 千: 1000 };
const BIG = { 萬: 1e4, 億: 1e8 };

const IDIOM_GUARD = [
  "千萬", "萬一", "萬分", "萬能", "萬歲", "萬全", "萬事", "萬象", "一二",
  "三三兩兩", "五花八門", "七上八下", "九牛一毛", "五湖四海", "十全十美",
  "亂七八糟", "十之八九", "一心一意", "三心二意", "四面八方", "百分之百",
  "一五一十", "三言兩語", "百年", "千年", "萬年", "一一", "十字", "百般",
  "千方百計", "成千上萬", "億萬",
];
const SKIP_SINGLE = new Set("零十百千萬億兩一二三四五六七八九");
const NUM_RE = /[零〇一二三四五六七八九十百千萬億兩]+/g;
const PCT_RE = /百分之([零〇一二三四五六七八九十百千點兩0-9]+)/g;

/* parse a section below 萬 (handles 十/百/千 place values) */
function parseSection(s) {
  let total = 0, cur = 0, seen = false;
  for (const ch of s) {
    if (ch in DIGIT) { cur = DIGIT[ch]; seen = true; }
    else if (ch in UNIT) {
      total += (cur || 1) * UNIT[ch]; cur = 0; seen = true;
    } else return null;
  }
  return seen ? total + cur : null;
}

/* full zh integer/decimal (with 點) up to 億 */
function zhToNumber(s) {
  const dot = s.indexOf("點");
  let intPart = s, frac = "";
  if (dot >= 0) {
    intPart = s.slice(0, dot);
    for (const ch of s.slice(dot + 1)) {
      if (!(ch in DIGIT)) return null;
      frac += DIGIT[ch];
    }
  }
  // split on 億 then 萬
  let value = 0;
  let rest = intPart;
  for (const [mark, mult] of [["億", BIG.億], ["萬", BIG.萬]]) {
    const i = rest.indexOf(mark);
    if (i >= 0) {
      const hi = rest.slice(0, i);
      const sec = hi === "" ? 1 : parseSection(hi);
      if (sec === null) return null;
      value += sec * mult;
      rest = rest.slice(i + 1);
    }
  }
  if (rest) {
    // bare digit string like 一二三 (no units) reads as concatenated digits
    if ([...rest].every((c) => c in DIGIT) && rest.length > 1 &&
        !/[十百千]/.test(rest)) {
      value += Number([...rest].map((c) => DIGIT[c]).join(""));
    } else {
      const sec = parseSection(rest);
      if (sec === null) return null;
      value += sec;
    }
  }
  if (frac) return Number(value + "." + frac);
  return value;
}

function guarded(text, start, end) {
  const window = text.slice(Math.max(0, start - 3), end + 3);
  const frag = text.slice(start, end);
  return IDIOM_GUARD.some((idi) => window.includes(idi) && idi.includes(frag));
}

export function itn(text) {
  if (!text) return text;
  // percentages first (but not the idiom 百分之百 or a bare single unit char,
  // matching the Python reference where cn2an rejects a lone 百/十)
  text = text.replace(PCT_RE, (m, g) => {
    if (IDIOM_GUARD.includes(m) || (g.length === 1 && SKIP_SINGLE.has(g))) return m;
    const v = zhToNumber(g);
    return v === null || !isFinite(v) ? m : `${v}%`;
  });
  // standalone number runs
  let out = "", last = 0, m;
  NUM_RE.lastIndex = 0;
  while ((m = NUM_RE.exec(text)) !== null) {
    const s = m.index, e = s + m[0].length, run = m[0];
    out += text.slice(last, s);
    last = e;
    if ((run.length === 1 && SKIP_SINGLE.has(run)) || guarded(text, s, e)) {
      out += run;
      continue;
    }
    const v = zhToNumber(run);
    out += (v === null || !isFinite(v)) ? run : String(v);
  }
  out += text.slice(last);
  return out;
}
