// components/layout/AppSidebar.vue
<template>
  <el-aside class="app-sidebar" width="240px">
    <div class="logo-container">
      <img src="@/assets/logo.svg" alt="Logo" class="logo" />
    </div>
    
    <el-menu
      :default-active="activeRoute"
      class="sidebar-menu"
      :router="true"
      @select="handleSelect"
    >
      <el-menu-item 
        v-for="item in menuItems" 
        :key="item.path"
        :index="item.path"
        @mouseenter="() => $emit('menuHover', item.color)"
      >
        <el-icon><component :is="item.icon" /></el-icon>
        <template #title>
          <span>{{ item.title }}</span>
        </template>
      </el-menu-item>
    </el-menu>
  </el-aside>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const props = defineProps({
  menuItems: {
    type: Array,
    required: true
  }
})

defineEmits(['menuHover'])

const route = useRoute()
const activeRoute = computed(() => route.path)

const handleSelect = (index) => {
  console.log('Selected menu item:', index)
}
</script>

<style scoped>
.app-sidebar {
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  height: 100vh;
  position: relative;
  z-index: 2;
}

.logo-container {
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.logo {
  height: 40px;
  object-fit: contain;
}

.sidebar-menu {
  border-right: none;
  background: transparent;
}

:deep(.el-menu-item) {
  color: #ffffff;
  height: 56px;
  line-height: 56px;
}

:deep(.el-menu-item:hover) {
  background: rgba(255, 255, 255, 0.1);
}

:deep(.el-menu-item.is-active) {
  background: rgba(255, 255, 255, 0.15);
}

:deep(.el-menu-item .el-icon) {
  color: #ffffff;
}
</style>