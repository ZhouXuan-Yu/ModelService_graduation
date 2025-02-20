<template>
  <AppLayout>
    <router-view></router-view>
  </AppLayout>
</template>

<script setup>
import 'element-plus/dist/index.css'
import AppLayout from '@/components/layout/AppLayout.vue'
</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  background: #000000;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app {
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  color: #ffffff;
}

/* Element Plus Dark Theme Overrides */
:root {
  --el-color-primary: #00ff9d;
  --el-color-success: #67c23a;
  --el-color-warning: #e6a23c;
  --el-color-danger: #f56c6c;
  --el-color-info: #909399;
  
  --el-border-color: rgba(255, 255, 255, 0.1);
  --el-border-color-light: rgba(255, 255, 255, 0.05);
  --el-border-color-lighter: rgba(255, 255, 255, 0.02);
  
  --el-text-color-primary: #ffffff;
  --el-text-color-regular: rgba(255, 255, 255, 0.8);
  --el-text-color-secondary: rgba(255, 255, 255, 0.6);
  --el-text-color-placeholder: rgba(255, 255, 255, 0.3);
  
  --el-bg-color: #1a1a1a;
  --el-bg-color-overlay: #242424;
  --el-fill-color-blank: transparent;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}

/* Element Plus Component Overrides */
.el-button {
  border-radius: 6px;
}

.el-input__wrapper,
.el-textarea__wrapper {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}

.el-input__wrapper:hover,
.el-textarea__wrapper:hover {
  border-color: rgba(255, 255, 255, 0.2) !important;
}

.el-input__wrapper.is-focus,
.el-textarea__wrapper.is-focus {
  border-color: var(--el-color-primary) !important;
  box-shadow: 0 0 0 1px var(--el-color-primary) !important;
}

.el-input__inner,
.el-textarea__inner {
  color: #ffffff !important;
  background: transparent !important;
}

.el-input__inner::placeholder,
.el-textarea__inner::placeholder {
  color: rgba(255, 255, 255, 0.3) !important;
}

/* Select Dropdown */
.el-select-dropdown {
  background: rgba(30, 35, 40, 0.95) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  backdrop-filter: blur(10px);
}

.el-select-dropdown__item {
  color: rgba(255, 255, 255, 0.8) !important;
}

.el-select-dropdown__item.hover,
.el-select-dropdown__item:hover {
  background: rgba(255, 255, 255, 0.1) !important;
}

.el-select-dropdown__item.selected {
  background: rgba(255, 255, 255, 0.15) !important;
  color: var(--el-color-primary) !important;
}

/* Dialog */
.el-dialog {
  background: rgba(30, 35, 40, 0.95) !important;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  backdrop-filter: blur(10px);
}

.el-dialog__header {
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.el-dialog__footer {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* Message */
.el-message {
  background: rgba(0, 0, 0, 0.8) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  backdrop-filter: blur(10px);
}

.el-message--success {
  border-color: var(--el-color-success) !important;
}

.el-message--warning {
  border-color: var(--el-color-warning) !important;
}

.el-message--error {
  border-color: var(--el-color-danger) !important;
}

/* Loading */
.el-loading-mask {
  background-color: rgba(0, 0, 0, 0.8) !important;
  backdrop-filter: blur(10px);
}

/* Radio */
.el-radio {
  --el-radio-text-color: rgba(255, 255, 255, 0.8);
  --el-radio-input-border-color: rgba(255, 255, 255, 0.2);
  margin-right: 20px;
}

.el-radio__label {
  color: var(--el-radio-text-color) !important;
}

/* Table */
.el-table {
  --el-table-border-color: rgba(255, 255, 255, 0.1);
  --el-table-header-bg-color: rgba(255, 255, 255, 0.05);
  --el-table-row-hover-bg-color: rgba(255, 255, 255, 0.1);
  background: transparent !important;
}

.el-table th,
.el-table tr,
.el-table td {
  background: transparent !important;
  border-bottom-color: var(--el-table-border-color) !important;
  color: rgba(255, 255, 255, 0.8) !important;
}

.el-table th {
  background: var(--el-table-header-bg-color) !important;
}

.el-table--enable-row-hover .el-table__body tr:hover > td {
  background: var(--el-table-row-hover-bg-color) !important;
}

/* Empty */
.el-empty__description {
  color: rgba(255, 255, 255, 0.4) !important;
}

/* Dark Theme Input Styles */
:deep(.el-input__wrapper),
:deep(.el-textarea__wrapper) {
  background-color: rgba(0, 0, 0, 0.2) !important;
  box-shadow: none !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

:deep(.el-input__inner),
:deep(.el-textarea__inner) {
  color: #ffffff !important;
  background: transparent !important;
}

:deep(.el-input__inner::placeholder),
:deep(.el-textarea__inner::placeholder) {
  color: rgba(255, 255, 255, 0.3) !important;
}

:deep(.el-input.is-disabled .el-input__inner),
:deep(.el-textarea.is-disabled .el-textarea__inner) {
  background-color: rgba(255, 255, 255, 0.04) !important;
  color: rgba(255, 255, 255, 0.3) !important;
}

/* Select Styles */
:deep(.el-select .el-input__wrapper) {
  background-color: rgba(0, 0, 0, 0.2) !important;
}

:deep(.el-select-dropdown) {
  background: rgba(30, 35, 40, 0.95) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  backdrop-filter: blur(10px);
}

:deep(.el-select-dropdown__item) {
  color: rgba(255, 255, 255, 0.8) !important;
}

:deep(.el-select-dropdown__item.hover),
:deep(.el-select-dropdown__item:hover) {
  background: rgba(255, 255, 255, 0.1) !important;
}

:deep(.el-select-dropdown__item.selected) {
  background: rgba(255, 255, 255, 0.15) !important;
  color: var(--el-color-primary) !important;
}

/* Form Styles */
:deep(.el-form-item__label) {
  color: rgba(255, 255, 255, 0.8) !important;
}

/* Button Styles */
:deep(.el-button) {
  background: rgba(255, 255, 255, 0.1) !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  color: #ffffff !important;
}

:deep(.el-button:hover) {
  background: rgba(255, 255, 255, 0.2) !important;
  border-color: rgba(255, 255, 255, 0.3) !important;
}

:deep(.el-button--primary) {
  background: var(--el-color-primary) !important;
  border-color: var(--el-color-primary) !important;
}

:deep(.el-button--primary:hover) {
  background: var(--el-color-primary-light-3) !important;
  border-color: var(--el-color-primary-light-3) !important;
}

/* Dialog Styles */
:deep(.el-dialog) {
  background: rgba(30, 35, 40, 0.95) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

:deep(.el-dialog__title) {
  color: #ffffff !important;
}

/* Table Styles */
:deep(.el-table) {
  background: transparent !important;
  color: #ffffff !important;
}

:deep(.el-table th),
:deep(.el-table tr),
:deep(.el-table td) {
  background: transparent !important;
  border-bottom-color: rgba(255, 255, 255, 0.1) !important;
  color: #ffffff !important;
}

:deep(.el-table--enable-row-hover .el-table__body tr:hover > td) {
  background: rgba(255, 255, 255, 0.1) !important;
}

/* Tabs Styles */
:deep(.el-tabs__item) {
  color: rgba(255, 255, 255, 0.6) !important;
}

:deep(.el-tabs__item.is-active) {
  color: var(--el-color-primary) !important;
}

:deep(.el-tabs__active-bar) {
  background-color: var(--el-color-primary) !important;
}

/* Card Styles */
:deep(.el-card) {
  background: rgba(30, 35, 40, 0.95) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #ffffff !important;
}

/* Radio Styles */
:deep(.el-radio__label) {
  color: rgba(255, 255, 255, 0.8) !important;
}

:deep(.el-radio__input.is-checked + .el-radio__label) {
  color: var(--el-color-primary) !important;
}

/* Empty State */
:deep(.el-empty__description) {
  color: rgba(255, 255, 255, 0.4) !important;
}
</style>