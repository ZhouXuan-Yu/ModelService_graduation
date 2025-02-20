import { defineStore } from 'pinia'

export const useLayoutStore = defineStore('layout', {
  state: () => ({
    activeColor: '#00ff9d',
    notifications: [],
    showTutorial: false
  }),
  
  actions: {
    setActiveColor(color) {
      this.activeColor = color
    },
    
    addNotification(notification) {
      this.notifications.push(notification)
    },
    
    removeNotification(id) {
      const index = this.notifications.findIndex(n => n.id === id)
      if (index !== -1) {
        this.notifications.splice(index, 1)
      }
    },
    
    setShowTutorial(show) {
      this.showTutorial = show
    }
  }
}) 