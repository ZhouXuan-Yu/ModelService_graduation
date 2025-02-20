// components/layout/AppLayout.vue
<template>
  <div class="app-wrapper" @mousemove="handleMouseMove">
    <!-- Dynamic Background Component -->
    <DynamicBackground 
      :mousePosition="mousePosition"
      :activeColor="activeColor"
    />
    
    <!-- Main Layout -->
    <el-container class="layout-container">
      <!-- Sidebar Component -->
      <AppSidebar 
        :menuItems="menuItems"
        @menuHover="handleMenuHover"
      />
      
      <!-- Main Content Area -->
      <el-container class="main-container">
        <!-- Header Component -->
        <AppHeader 
          @buttonHover="handleButtonHover"
          @showNotification="handleNotification"
        />
        
        <!-- Main Content with Transitions -->
        <el-main class="main-content">
          <slot></slot>
        </el-main>
      </el-container>
    </el-container>

    <!-- Global Notification System -->
    <NotificationSystem ref="notificationSystem" />
    
    <!-- Interactive Tutorial -->
    <TutorialOverlay 
      v-if="showTutorial"
      @close="closeTutorial"
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { gsap } from 'gsap'
import DynamicBackground from './DynamicBackground.vue'
import AppSidebar from './AppSidebar.vue'
import AppHeader from './AppHeader.vue'
import NotificationSystem from './NotificationSystem.vue'
import TutorialOverlay from './TutorialOverlay.vue'

// State
const mousePosition = ref({ x: 0, y: 0 })
const activeColor = ref('#00ff9d')
const showTutorial = ref(false)
const notificationSystem = ref(null)

// Menu Items Configuration
const menuItems = [
  { 
    path: '/route-planning', 
    icon: 'Location', 
    title: '智能路径规划',
    color: '#00ff9d',
    description: '基于AI的智能路径规划系统'
  },
  { 
    path: '/image-recognition', 
    icon: 'Picture', 
    title: '高级图像识别',
    color: '#00d6ff',
    description: '先进的计算机视觉识别系统'
  },
  //实时追踪先取消掉
  // { 
  //   path: '/video-tracking', 
  //   icon: 'VideoCamera', 
  //   title: '实时视频追踪',
  //   color: '#ff00e5',
  //   description: '精确的目标追踪分析系统'
  // }
]

// Event Handlers
const handleMouseMove = (e) => {
  mousePosition.value = {
    x: e.clientX / window.innerWidth,
    y: e.clientY / window.innerHeight
  }
}

const handleMenuHover = (color) => {
  activeColor.value = color
}

const handleButtonHover = (color) => {
  activeColor.value = color
}

const handleNotification = (notification) => {
  notificationSystem.value?.addNotification(notification)
}

const closeTutorial = () => {
  showTutorial.value = false
}

// Page Transition Animations
const beforeEnter = (el) => {
  gsap.set(el, {
    opacity: 0,
    y: 50
  })
}

const enter = (el, done) => {
  gsap.to(el, {
    opacity: 1,
    y: 0,
    duration: 0.6,
    ease: 'power2.out',
    onComplete: done
  })
}

const leave = (el, done) => {
  gsap.to(el, {
    opacity: 0,
    y: -50,
    duration: 0.4,
    ease: 'power2.in',
    onComplete: done
  })
}

// Lifecycle
onMounted(() => {
  const hasSeenTutorial = localStorage.getItem('hasSeenTutorial')
  if (!hasSeenTutorial) {
    showTutorial.value = true
    localStorage.setItem('hasSeenTutorial', 'true')
  }
})

// Expose methods
defineExpose({
  showNotification: (notification) => {
    notificationSystem.value?.addNotification(notification)
  }
})
</script>

<style scoped>
.app-wrapper {
  position: relative;
  min-height: 100vh;
  overflow: hidden;
  background-color: #000000;
}

.layout-container {
  position: relative;
  height: 100vh;
  z-index: 1;
}

.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.main-content {
  flex: 1;
  padding: 32px;
  overflow-y: auto;
  position: relative;
}

/* Custom Scrollbar */
.main-content::-webkit-scrollbar {
  width: 8px;
}

.main-content::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
}

.main-content::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}

.main-content::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}

/* Page Transition Classes */
.page-transition-enter-active,
.page-transition-leave-active {
  position: absolute;
  width: 100%;
}

.page-transition-enter-from,
.page-transition-leave-to {
  opacity: 0;
  transform: translateY(30px);
}
</style>