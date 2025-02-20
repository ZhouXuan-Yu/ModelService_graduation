import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAnalysisStore = defineStore('analysis', () => {
  // 状态
  const analysisResult = ref(null)
  const activePersonId = ref(null)
  const queryHistory = ref([])

  // 计算属性
  const hasResults = computed(() => analysisResult.value !== null)
  const activePerson = computed(() => {
    if (!activePersonId.value || !analysisResult.value) return null
    return analysisResult.value.faces_info[activePersonId.value]
  })

  // 方法
  function setAnalysisResult(result) {
    analysisResult.value = result
  }

  function setActivePerson(id) {
    activePersonId.value = id
  }

  function addQueryHistory(query) {
    queryHistory.value.push({
      query,
      timestamp: new Date().toISOString()
    })
  }

  function findMatchingPersons(conditions) {
    if (!analysisResult.value) return []

    return analysisResult.value.faces_info.map((person, index) => ({
      ...person,
      id: index
    })).filter(person => {
      // 性别匹配
      if (conditions.gender && person.gender !== conditions.gender) {
        return false
      }

      // 年龄匹配
      if (conditions.age) {
        const age = parseFloat(person.age)
        if (conditions.age.includes('<') && age >= parseFloat(conditions.age.slice(1))) {
          return false
        }
        if (conditions.age.includes('>') && age <= parseFloat(conditions.age.slice(1))) {
          return false
        }
      }

      // 上衣颜色匹配
      if (conditions.upperColor && person.upper_color !== conditions.upperColor) {
        return false
      }

      // 下装颜色匹配
      if (conditions.lowerColor && person.lower_color !== conditions.lowerColor) {
        return false
      }

      return true
    })
  }

  function clearResults() {
    analysisResult.value = null
    activePersonId.value = null
  }

  return {
    // 状态
    analysisResult,
    activePersonId,
    queryHistory,
    // 计算属性
    hasResults,
    activePerson,
    // 方法
    setAnalysisResult,
    setActivePerson,
    addQueryHistory,
    findMatchingPersons,
    clearResults
  }
}) 