// components/layout/DynamicBackground.vue
<template>
  <div class="dynamic-background">
    <div 
      class="gradient-overlay"
      :style="{
        background: `radial-gradient(
          circle at ${mousePosition.x * 100}% ${mousePosition.y * 100}%,
          ${activeColor}33 0%,
          transparent 50%
        )`
      }"
    ></div>
  </div>
</template>

<script setup>
defineProps({
  mousePosition: {
    type: Object,
    default: () => ({ x: 0.5, y: 0.5 })
  },
  activeColor: {
    type: String,
    default: '#00ff9d'
  }
})
</script>

<style scoped>
.dynamic-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 0;
  pointer-events: none;
}

.gradient-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  transition: background 0.3s ease;
}
</style>