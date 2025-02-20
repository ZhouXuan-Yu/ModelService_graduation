import { createRouter, createWebHistory } from 'vue-router'

// 使用动态导入
const HomeView = () => import('../views/HomeView.vue')
const RoutePlanningView = () => import('../views/RoutePlanningView.vue')
const ImageRecognitionView = () => import('../views/ImageRecognitionView.vue')
const VideoTrackingView = () => import('../views/VideoTrackingView.vue')

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/route-planning',
      name: 'route-planning',
      component: RoutePlanningView
    },
    {
      path: '/image-recognition',
      name: 'image-recognition',
      component: ImageRecognitionView
    },
    {
      path: '/video-tracking',
      name: 'video-tracking',
      component: VideoTrackingView
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/'
    }
  ]
})

// 添加路由守卫
router.beforeEach((to, from, next) => {
  if (!to.matched.length) {
    next({ name: 'home' })
    return
  }
  next()
})

export default router 