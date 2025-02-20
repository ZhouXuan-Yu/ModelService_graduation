import { defineStore } from 'pinia'
import { routePlanningApi } from '@/api/routePlanning'

export const useRoutePlanningStore = defineStore('routePlanning', {
  state: () => ({
    history: [],
    favorites: [],
    currentRoute: null
  }),

  actions: {
    async getRoutePlan(data) {
      try {
        const result = await routePlanningApi.getRoutePlan({
          text: data.text,
          model: data.model,
          signal: data.signal
        })
        if (result.route_id) {
          this.currentRoute = {
            id: result.route_id,
            ...result
          }
        }
        return result
      } catch (error) {
        throw error
      }
    },

    async getHistory() {
      try {
        const result = await routePlanningApi.getHistory()
        this.history = result
        return result
      } catch (error) {
        console.error('获取历史记录失败:', error)
        this.history = []
        throw error
      }
    },

    async deleteHistory(routeId) {
      try {
        await routePlanningApi.deleteHistory(routeId)
        this.history = this.history.filter(item => item.id !== routeId)
        return true
      } catch (error) {
        console.error('删除历史记录失败:', error)
        throw error
      }
    },

    async getFavorites() {
      try {
        const result = await routePlanningApi.getFavorites()
        this.favorites = result
        return result
      } catch (error) {
        console.error('获取收藏失败:', error)
        this.favorites = []
        throw error
      }
    },

    async addFavorite(data) {
      try {
        const result = await routePlanningApi.addFavorite(data)
        this.favorites.unshift(result)
        return result
      } catch (error) {
        console.error('添加收藏失败:', error)
        throw error
      }
    },

    async removeFavorite(routeId) {
      try {
        await routePlanningApi.removeFavorite(routeId)
        this.favorites = this.favorites.filter(item => item.id !== routeId)
        return true
      } catch (error) {
        console.error('取消收藏失败:', error)
        throw error
      }
    },

    async exportRoute(data) {
      try {
        return await routePlanningApi.exportRoute(data)
      } catch (error) {
        console.error('导出路线失败:', error)
        throw error
      }
    },

    async getRouteDetail(routeId) {
      try {
        const result = await routePlanningApi.getRouteDetail(routeId)
        return result
      } catch (error) {
        console.error('获取路线详情失败:', error)
        throw error
      }
    }
  }
}) 