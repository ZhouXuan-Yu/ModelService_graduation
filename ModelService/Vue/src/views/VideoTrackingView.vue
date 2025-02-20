<template>
  <div class="video-tracking-view">
    <VideoTracking />
  </div>
</template>

<script setup>
import VideoTracking from './VideoTracking.vue'
</script>

<style scoped>
.video-tracking-view {
  height: 100%;
  width: 100%;
  overflow-y: auto;
  padding: 20px;
}
</style> 