/**
 * Redux Store 配置
 * 使用Redux Toolkit创建store
 */

import { configureStore } from '@reduxjs/toolkit'
import authSlice from './authSlice'
import datasetSlice from './datasetSlice'
import modelSlice from './modelSlice'
import trainingSlice from './trainingSlice'
import inferenceSlice from './inferenceSlice'
import uiSlice from './uiSlice'

export const store = configureStore({
  reducer: {
    auth: authSlice,
    dataset: datasetSlice,
    model: modelSlice,
    training: trainingSlice,
    inference: inferenceSlice,
    ui: uiSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [
          'persist/PERSIST',
          'persist/REHYDRATE',
          'persist/PAUSE',
          'persist/FLUSH',
          'persist/REGISTER',
          'persist/REVERT'
        ],
        ignoredPaths: ['ui.modals.confirmDialog']
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
})

// 类型导出，如果项目升级到TypeScript时使用
// export type RootState = ReturnType<typeof store.getState>
// export type AppDispatch = typeof store.dispatch

export default store