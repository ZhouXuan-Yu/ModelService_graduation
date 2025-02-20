// 英文到中文的颜色映射
export const COLOR_EN_TO_CN = {
  'red': '红色',
  'orange': '橙色',
  'yellow': '黄色',
  'green': '绿色',
  'blue': '蓝色',
  'purple': '紫色',
  'pink': '粉色',
  'brown': '棕色',
  'black': '黑色',
  'white': '白色',
  'gray': '灰色',
  'navy': '藏青色',
  'beige': '米色',
  'khaki': '卡其色'
}

// 中文到英文的颜色映射
export const COLOR_CN_TO_EN = Object.entries(COLOR_EN_TO_CN).reduce((acc, [en, cn]) => {
  // 同时支持带"色"和不带"色"的中文颜色名
  acc[cn] = en
  acc[cn.replace('色', '')] = en
  return acc
}, {})

// 颜色样式映射
export const COLOR_STYLES = {
  'red': { background: '#ffebee', text: '#d32f2f' },
  'orange': { background: '#fff3e0', text: '#ef6c00' },
  'yellow': { background: '#fffde7', text: '#f57f17' },
  'green': { background: '#e8f5e9', text: '#2e7d32' },
  'blue': { background: '#e3f2fd', text: '#1976d2' },
  'purple': { background: '#f3e5f5', text: '#7b1fa2' },
  'pink': { background: '#fce4ec', text: '#c2185b' },
  'brown': { background: '#efebe9', text: '#5d4037' },
  'black': { background: '#fafafa', text: '#212121' },
  'white': { background: '#ffffff', text: '#757575' },
  'gray': { background: '#f5f5f5', text: '#616161' },
  'navy': { background: '#e8eaf6', text: '#283593' },
  'beige': { background: '#fff8e1', text: '#8d6e63' },
  'khaki': { background: '#f0f4c3', text: '#827717' }
}

/**
 * 获取颜色的样式
 * @param {string} color - 颜色名称（英文）
 * @returns {Object} - 颜色样式对象
 */
export function getColorStyle(color) {
  const style = COLOR_STYLES[color] || COLOR_STYLES.gray
  return {
    backgroundColor: style.background,
    color: style.text,
    padding: '2px 8px',
    borderRadius: '4px',
    border: `1px solid ${style.text}20`,
    fontWeight: 500
  }
}

// 相似颜色组
export const COLOR_GROUPS = {
  'red': ['red', 'pink'],
  'orange': ['orange', 'brown'],
  'yellow': ['yellow', 'beige'],
  'green': ['green'],
  'blue': ['blue', 'navy'],
  'purple': ['purple'],
  'brown': ['brown', 'khaki'],
  'black': ['black', 'gray'],
  'white': ['white', 'beige'],
  'gray': ['gray']
}

/**
 * 获取相似颜色列表
 * @param {string} color - 颜色名称（英文）
 * @returns {Array} - 相似颜色列表（英文）
 */
export function getSimilarColors(color) {
  for (const [mainColor, group] of Object.entries(COLOR_GROUPS)) {
    if (group.includes(color)) {
      return group
    }
  }
  return [color]
}

/**
 * 判断两个颜色是否相似
 * @param {string} color1 - 第一个颜色（英文）
 * @param {string} color2 - 第二个颜色（英文）
 * @returns {boolean} - 是否相似
 */
export function isSimilarColor(color1, color2) {
  const similarColors = getSimilarColors(color1)
  return similarColors.includes(color2)
}

/**
 * 翻译颜色名称
 * @param {string} color - 颜色名称
 * @param {boolean} toChinese - 是否翻译成中文
 * @returns {string} - 翻译后的颜色名称
 */
export function translateColor(color, toChinese = true) {
  if (!color) return ''
  
  if (toChinese) {
    return COLOR_EN_TO_CN[color] || color
  } else {
    return COLOR_CN_TO_EN[color] || color
  }
} 