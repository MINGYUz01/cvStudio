/**
 * Redux Store 配置
 * 使用Redux Toolkit创建store
 */

import { configureStore } from '@reduxjs/toolkit'
import authSlice from './authSlice'
import datasetSlice from './datasetSlice'
import modelSlice from './modelSlice'

export const store = configureStore({
  reducer: {
    auth: authSlice,
    dataset: datasetSlice,
    model: modelSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
})

export default store