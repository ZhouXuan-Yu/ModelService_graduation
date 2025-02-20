<template>
  <div class="guide-system">
    <!-- 添加悬浮帮助按钮 -->
    <el-button
      class="help-button"
      type="primary"
      circle
      @click="showWelcome = true"
    >
      <el-icon><QuestionFilled /></el-icon>
    </el-button>

    <!-- 首次使用欢迎弹窗 -->
    <el-dialog
      v-model="showWelcome"
      title="欢迎使用智能路线规划"
      width="800px"
      :show-close="true"
      :close-on-click-modal="true"
    >
      <div class="welcome-content">
        <p>让我来帮您快速了解系统的主要功能</p>
        
        <!-- 添加搜索框 -->
        <div class="scene-search">
          <el-input
            v-model="searchQuery"
            placeholder="搜索场景..."
            prefix-icon="Search"
            clearable
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </div>

        <!-- 场景分类展示 -->
        <div class="scene-categories">
          <el-tabs type="border-card">
            <el-tab-pane 
              v-for="category in filteredCategories" 
              :key="category.id"
              :label="category.name"
            >
              <template #label>
                <el-icon><component :is="category.icon" /></el-icon>
                <span>{{ category.name }}</span>
              </template>
              
              <div class="category-grid">
                <el-card
                  v-for="action in category.actions"
                  :key="action.id"
                  class="action-card"
                  :class="{ 'is-active': selectedAction === action.id }"
                  @click="startQuickAction(action)"
                >
                  <div class="action-content">
                    <div class="action-icon-wrapper">
                      <el-icon class="action-icon">
                        <component :is="action.icon" />
                      </el-icon>
                    </div>
                    <h3>{{ action.title }}</h3>
                    <p>{{ action.description }}</p>
                    <!-- 添加标签 -->
                    <div class="action-tags">
                      <el-tag 
                        v-for="tag in action.tags" 
                        :key="tag"
                        size="small"
                        :type="getTagType(tag)"
                      >
                        {{ tag }}
                      </el-tag>
                    </div>
                  </div>
                </el-card>
              </div>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="skipGuide">跳过</el-button>
          <el-button type="primary" @click="startGuide">开始向导</el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 功能引导提示 -->
    <el-popover
      v-model:visible="showGuide"
      :visible="showGuide"
      :width="300"
      trigger="manual"
      :popper-style="{ padding: '16px' }"
      :virtual-ref="currentTarget"
      virtual-triggering
      placement="bottom"
    >
      <template #default>
        <div class="guide-step">
          <h3>{{ currentStep.title }}</h3>
          <p>{{ currentStep.content }}</p>
          <div class="guide-actions">
            <el-button size="small" @click="prevStep" :disabled="stepIndex === 0">
              上一步
            </el-button>
            <el-button
              size="small"
              type="primary"
              @click="nextStep"
            >
              {{ stepIndex === guideSteps.length - 1 ? '���成' : '下一步' }}
            </el-button>
          </div>
        </div>
      </template>
    </el-popover>

    <!-- 智能提示 -->
    <div v-if="showSuggestions" class="input-suggestions">
      <div
        v-for="(suggestion, index) in currentSuggestions"
        :key="index"
        class="suggestion-item"
        @click="applySuggestion(suggestion)"
      >
        {{ suggestion }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, onBeforeUnmount } from 'vue'
import { useLocalStorage } from '@vueuse/core'
import { 
  Van, Briefcase, Money, Place, Timer,
  Location, Position, Connection, Sunny,
  Ship, Bicycle, Trophy, Calendar, Search,
  QuestionFilled
} from '@element-plus/icons-vue'

// 场景分类配置
const categories = [
  {
    id: 'daily',
    name: '日常出行',
    icon: 'Van',
    actions: [
      {
        id: 'commute',
        title: '通勤路线',
        description: '智能规划上下班路线',
        icon: 'Briefcase',
        prompt: '帮我规划一条从 {start} 到 {end} 的通勤路线，考虑实时路况',
        tags: ['实时路况', '高峰避堵']
      },
      {
        id: 'economic',
        title: '经济路线',
        description: '优先选择免费道路',
        icon: 'Money',
        prompt: '帮我规划一条从 {start} 到 {end} 的经济路线，尽量避免收费',
        tags: ['省钱', '避免收费']
      }
    ]
  },
  {
    id: 'leisure',
    name: '休闲娱乐',
    icon: 'Sunny',
    actions: [
      {
        id: 'tourism',
        title: '景点游览',
        description: '合理规划游览顺序',
        icon: 'Place',
        prompt: '帮我规划景点游览路线，包含以下景点: {spots}',
        tags: ['景点']
      },
      {
        id: 'weekend',
        title: '周末游玩',
        description: '轻松愉快的休闲路线',
        icon: 'Calendar',
        prompt: '帮我规划一条周末休闲路线，包含有趣的地方',
        tags: ['休闲']
      }
    ]
  },
  {
    id: 'business',
    name: '商务出行',
    icon: 'Briefcase',
    actions: [
      {
        id: 'meeting',
        title: '会议行程',
        description: '准时到达会议地点',
        icon: 'Timer',
        prompt: '帮我规划去 {destination} 参加会议的路线，需要 {time} 前到达',
        tags: ['紧急']
      },
      {
        id: 'airport',
        title: '机场接送',
        description: '往返机场路线',
        icon: 'Ship',
        prompt: '帮我规划从 {start} 到机场的路线，需要考虑航班时间',
        tags: ['���急']
      }
    ]
  },
  {
    id: 'special',
    name: '特殊场景',
    icon: 'Trophy',
    actions: [
      {
        id: 'emergency',
        title: '紧急路线',
        description: '最快到达目的地',
        icon: 'Timer',
        prompt: '帮我规划一条从 {start} 到 {end} 的紧急路线，必须最快到达',
        tags: ['紧急']
      },
      {
        id: 'scenic',
        title: '风景路线',
        description: '沿途欣赏美景',
        icon: 'Sunny',
        prompt: '帮我规划一条风景优美的路线，可以欣赏沿途景色',
        tags: ['风景']
      }
    ]
  }
]

// 引导步骤配置
const guideSteps = [
  {
    target: '.chat-input',
    title: '智能对话',
    content: '您可以直接描述您的出行需求，比如"帮我规划一条从京西站到首都机场的路线"'
  },
  {
    target: '.quick-actions',
    title: '快速操作',
    content: '点击这些卡片可以快速开始常用场景的路线规划'
  },
  {
    target: '.layer-control',
    title: '图层控制',
    content: '在这里可以切换地图显示模式，包括实时路况、卫星图像等'
  }
]

// 状态管理
const isFirstVisit = useLocalStorage('isFirstVisit', true)
const showWelcome = ref(false)
const showGuide = ref(false)
const stepIndex = ref(0)
const currentTarget = ref(null)
const showSuggestions = ref(false)
const currentSuggestions = ref([])
const selectedAction = ref(null)

// 计算当前步骤
const currentStep = computed(() => guideSteps[stepIndex.value])

// 方法
const skipGuide = () => {
  showWelcome.value = false
  isFirstVisit.value = false
}

const startGuide = () => {
  showWelcome.value = false
  showGuide.value = true
  stepIndex.value = 0
  updateTarget()
}

const updateTarget = () => {
  const targetElement = document.querySelector(currentStep.value.target)
  if (targetElement) {
    currentTarget.value = targetElement
  }
}

const nextStep = () => {
  if (stepIndex.value < guideSteps.length - 1) {
    stepIndex.value++
    updateTarget()
  } else {
    showGuide.value = false
    isFirstVisit.value = false
  }
}

const prevStep = () => {
  if (stepIndex.value > 0) {
    stepIndex.value--
    updateTarget()
  }
}

const startQuickAction = (action) => {
  selectedAction.value = action.id
  
  // 添加动画延迟
  setTimeout(() => {
    emit('quick-action', action)
  }, 300)
}

const applySuggestion = (suggestion) => {
  // 应用输入建议
  emit('apply-suggestion', suggestion)
  showSuggestions.value = false
}

// 添加场景过滤功能
const searchQuery = ref('')
const filteredCategories = computed(() => {
  if (!searchQuery.value) return categories
  
  return categories.map(category => ({
    ...category,
    actions: category.actions.filter(action => 
      action.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      action.description.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      action.tags?.some(tag => tag.toLowerCase().includes(searchQuery.value.toLowerCase()))
    )
  })).filter(category => category.actions.length > 0)
})

// 标签类型
const getTagType = (tag) => {
  const tagTypes = {
    '实时路况': 'success',
    '高峰避堵': 'warning',
    '省钱': 'info',
    '避免收费': 'info',
    '景点': 'danger',
    '休闲': 'success',
    '紧急': 'danger',
    '风景': 'success',
    '其他': 'info'
  }
  return tagTypes[tag] || 'info'
}

// 添加重置方法
const resetGuide = () => {
  localStorage.removeItem('isFirstVisit')
  showWelcome.value = true
}

// 暴露方法给父组件
defineExpose({
  showSuggestions: (suggestions) => {
    currentSuggestions.value = suggestions
    showSuggestions.value = true
  },
  hideSuggestions: () => {
    showSuggestions.value = false
  },
  resetGuide  // 暴露重置方法
})

// 事件
const emit = defineEmits(['quick-action', 'apply-suggestion'])

// 生命周期
onMounted(() => {
  if (isFirstVisit.value) {
    showWelcome.value = true
  }

  // 初始显示帮助按钮
  autoHideHelpButton()
  
  // 监听鼠标移动，在移动时显示按钮
  document.addEventListener('mousemove', autoHideHelpButton)
})

onBeforeUnmount(() => {
  document.removeEventListener('mousemove', autoHideHelpButton)
})

// 可以添加一个动画标志
const helpButtonVisible = ref(true)

// 添加自动隐藏和显示帮助按钮的逻辑
let hideTimeout
const autoHideHelpButton = () => {
  if (hideTimeout) clearTimeout(hideTimeout)
  helpButtonVisible.value = true
  hideTimeout = setTimeout(() => {
    helpButtonVisible.value = false
  }, 3000)  // 3秒后自动隐藏
}
</script>

<style scoped>
.guide-system {
  position: relative;
}

.welcome-content {
  text-align: center;
  padding: 20px 0;
}

.quick-start {
  margin-top: 24px;
}

.scene-categories {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.category-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.category-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 16px;
  color: var(--el-text-color-primary);
  padding-bottom: 8px;
  border-bottom: 1px solid var(--el-border-color-light);
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
}

.action-card {
  height: 100%;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid var(--el-border-color-light);
}

.action-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
  border-color: var(--el-color-primary-light-5);
}

.action-icon-wrapper {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: var(--el-color-primary-light-9);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 12px;
  transition: transform 0.3s ease;
}

.action-card:hover .action-icon-wrapper {
  transform: rotate(12deg) scale(1.1);
}

.action-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 16px;
}

.action-content h3 {
  margin: 0 0 8px;
  font-size: 16px;
  color: var(--el-text-color-primary);
}

.action-content p {
  margin: 0;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.4;
}

.guide-step {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.guide-step h3 {
  margin: 0;
  font-size: 16px;
  color: var(--el-color-primary);
}

.guide-step p {
  margin: 0;
  font-size: 14px;
  color: var(--el-text-color-regular);
}

.guide-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}

.input-suggestions {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: var(--el-bg-color);
  border-radius: 4px;
  box-shadow: var(--el-box-shadow-light);
  margin-top: 4px;
  z-index: 100;
}

.suggestion-item {
  padding: 8px 16px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.suggestion-item:hover {
  background-color: var(--el-color-primary-light-9);
}

/* 添加动画相关样式 */
.action-list-move,
.action-list-enter-active,
.action-list-leave-active {
  transition: all 0.5s ease;
}

.action-list-enter-from,
.action-list-leave-to {
  opacity: 0;
  transform: translateY(30px);
}

.action-list-leave-active {
  position: absolute;
}

.action-card {
  transform-origin: center;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.action-card:hover {
  transform: translateY(-5px) scale(1.02);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}

.action-card.is-active {
  border-color: var(--el-color-primary);
  box-shadow: 0 0 0 2px var(--el-color-primary-light-3);
}

.action-icon-wrapper {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: var(--el-color-primary-light-9);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 12px;
  transition: transform 0.3s ease;
}

.action-card:hover .action-icon-wrapper {
  transform: rotate(15deg);
}

.action-icon {
  font-size: 24px;
  color: var(--el-color-primary);
}

.scene-search {
  margin-bottom: 16px;
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  position: relative;
}

.scene-search {
  margin: 16px auto;
  max-width: 400px;
}

.scene-categories {
  margin-top: 24px;
}

:deep(.el-tabs__item) {
  display: flex;
  align-items: center;
  gap: 8px;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
  padding: 16px;
}

.action-tags {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: center;
}

/* 添加暗色主题适配 */
:deep(.el-tabs--border-card) {
  background: var(--el-bg-color);
  border-color: var(--el-border-color);
}

:deep(.el-tabs--border-card > .el-tabs__header) {
  background: var(--el-bg-color-overlay);
  border-bottom-color: var(--el-border-color);
}

:deep(.el-tabs--border-card > .el-tabs__header .el-tabs__item.is-active) {
  background: var(--el-color-primary-light-9);
  border-right-color: var(--el-border-color);
  border-left-color: var(--el-border-color);
}

.help-button {
  position: fixed;
  bottom: 40px;
  right: 40px;
  z-index: 2000;
  width: 48px;
  height: 48px;
  font-size: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
  opacity: 0.6;
}

.help-button:hover {
  transform: scale(1.1);
  opacity: 1;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

/* 添加呼吸灯效果 */
@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(var(--el-color-primary-rgb), 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(var(--el-color-primary-rgb), 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(var(--el-color-primary-rgb), 0);
  }
}

.help-button {
  animation: pulse 2s infinite;
}

/* 添加淡入淡出效果 */
.help-button-enter-active,
.help-button-leave-active {
  transition: opacity 0.3s ease;
}

.help-button-enter-from,
.help-button-leave-to {
  opacity: 0;
}

/* 确保弹窗在暗色主题下正常显示 */
:deep(.el-dialog) {
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-light);
}

:deep(.el-dialog__header) {
  border-bottom: 1px solid var(--el-border-color-light);
  margin-right: 0;
  padding-right: 20px;
}

:deep(.el-dialog__headerbtn) {
  font-size: 18px;
}

:deep(.el-dialog__title) {
  font-size: 18px;
  font-weight: 600;
}
</style> 