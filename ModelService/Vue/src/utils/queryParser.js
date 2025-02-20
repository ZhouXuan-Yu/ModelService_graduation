/**
 * 查询解析工具
 */

import { COLOR_CN_TO_EN, COLOR_EN_TO_CN } from './colorMapping'

// 性别关键词映射
const GENDER_KEYWORDS = {
  '女': '女',
  '女性': '女',
  '女生': '女',
  '男': '男',
  '男性': '男',
  '男生': '男'
}

// 颜色关键词正则
const COLOR_PATTERN = new RegExp(Object.keys(COLOR_CN_TO_EN).join('|'), 'g')

// 年龄关键词正则
const AGE_PATTERN = /(\d+)岁|年龄[大小于于](\d+)/g

/**
 * 解析自然语言查询
 * @param {string} query - 用户输入的查询字符串
 * @returns {Object} - 解析后的查询条件
 */
export function parseQuery(query) {
  const conditions = {}
  
  // 解析性别
  for (const [keyword, value] of Object.entries(GENDER_KEYWORDS)) {
    if (query.includes(keyword)) {
      conditions.gender = value
      break
    }
  }
  
  // 解析颜色
  const colorMatches = query.match(COLOR_PATTERN)
  if (colorMatches) {
    // 判断是上衣还是下装的颜色
    const lastColorMatch = colorMatches[colorMatches.length - 1]
    if (query.includes('上衣') || query.includes('上装') || query.includes('衬衫') || query.includes('外套')) {
      conditions.upperColor = COLOR_CN_TO_EN[lastColorMatch]
    } else if (query.includes('裤子') || query.includes('下装') || query.includes('裙子')) {
      conditions.lowerColor = COLOR_CN_TO_EN[lastColorMatch]
    } else {
      // 默认为上衣颜色
      conditions.upperColor = COLOR_CN_TO_EN[lastColorMatch]
    }
  }
  
  // 解析年龄
  const ageMatches = [...query.matchAll(AGE_PATTERN)]
  if (ageMatches.length > 0) {
    const match = ageMatches[0]
    const age = match[1] || match[2]
    if (query.includes('大于') || query.includes('超过')) {
      conditions.age = `>${age}`
    } else if (query.includes('小于') || query.includes('低于')) {
      conditions.age = `<${age}`
    } else {
      conditions.age = `=${age}`
    }
  }
  
  return conditions
}

/**
 * 生成查询结果描述
 * @param {Array} matches - 匹配的人物列表
 * @param {Object} conditions - 查询条件
 * @returns {string} - 生成的描述文本
 */
export function generateResultDescription(matches, conditions) {
  if (matches.length === 0) {
    return '抱歉，没有找到符合条件的人物。'
  }
  
  let description = `找到了 ${matches.length} 个符合条件的人物：\n\n`
  
  matches.forEach((person, index) => {
    description += `${index + 1}. `
    
    // 添加性别和年龄信息
    const gender = person.gender === 'female' ? '女性' : '男性'
    description += `${gender}，约${Math.round(person.age)}岁`
    
    // 添加衣着信息
    if (person.upper_color) {
      description += `，穿着${COLOR_EN_TO_CN[person.upper_color]}色上衣`
    }
    if (person.lower_color) {
      description += `，${COLOR_EN_TO_CN[person.lower_color]}色下装`
    }
    
    description += '\n'
  })
  
  return description
}

/**
 * 验证查询条件是否有效
 * @param {Object} conditions 查询条件
 * @returns {boolean} 是否有效
 */
export function validateConditions(conditions) {
  return Object.keys(conditions).length > 0
} 