/**
 * 系统设置页面组件 - 现代化设计
 */

import React, { useState } from 'react'
import {
  User,
  Settings as SettingsIcon,
  Shield,
  Database,
  HardDrive,
  Monitor,
  Bell,
  Globe,
  Moon,
  Sun,
  Palette,
  Save,
  RotateCcw,
  Download,
  Upload,
  Key,
  Lock,
  Mail,
  Smartphone,
  Wifi,
  Server,
  Cpu,
  Zap,
  Info,
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  HelpCircle,
  FileText,
  Trash2,
  Plus,
  Edit2,
  Eye,
  EyeOff,
  Copy,
  RefreshCw
} from 'lucide-react'

/**
 * 系统设置页面
 * 管理用户偏好、系统配置和高级选项
 */
const Settings = () => {
  const [activeTab, setActiveTab] = useState('profile')
  const [showPassword, setShowPassword] = useState(false)
  const [notifications, setNotifications] = useState({
    email: true,
    push: true,
    training: true,
    inference: false,
    system: true
  })
  const [theme, setTheme] = useState('dark')
  const [language, setLanguage] = useState('zh-CN')

  // 模拟用户数据
  const userData = {
    username: 'admin',
    email: 'admin@cvstudio.com',
    fullName: '系统管理员',
    role: 'Administrator',
    joinDate: '2024-01-01',
    lastLogin: '2024-01-15 16:30:00',
    avatar: null,
    status: 'active'
  }

  // 模拟系统配置数据
  const systemConfig = {
    version: 'v1.0.0',
    buildDate: '2024-01-15',
    environment: 'production',
    maxUsers: 100,
    maxDatasets: 500,
    maxModels: 200,
    storageQuota: '1TB',
    usedStorage: '458GB',
    gpuEnabled: true,
    gpuCount: 2,
    gpuType: 'RTX 4090'
  }

  /**
   * 设置项组件
   */
  const SettingItem = ({ icon: Icon, title, description, children, action }) => (
    <div className="flex items-center justify-between p-4 bg-slate-700/30 rounded-lg hover:bg-slate-700/50 transition-colors">
      <div className="flex items-center gap-3 flex-1">
        <div className="w-10 h-10 bg-slate-600/50 rounded-lg flex items-center justify-center">
          <Icon size={20} className="text-slate-300" />
        </div>
        <div className="flex-1">
          <h4 className="text-white font-medium">{title}</h4>
          <p className="text-slate-400 text-sm">{description}</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {children}
        {action && (
          <button className="px-3 py-1 text-sm bg-slate-600/50 text-slate-300 rounded hover:bg-slate-600/70 transition-colors">
            {action}
          </button>
        )}
      </div>
    </div>
  )

  /**
   * 开关组件
   */
  const ToggleSwitch = ({ checked, onChange, disabled = false }) => (
    <button
      onClick={() => onChange && onChange(!checked)}
      disabled={disabled}
      className={`relative w-12 h-6 rounded-full transition-colors ${
        checked ? 'bg-blue-500' : 'bg-slate-600'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    >
      <div
        className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
          checked ? 'translate-x-7' : 'translate-x-1'
        }`}
      ></div>
    </button>
  )

  /**
   * 输入框组件
   */
  const SettingInput = ({ type, value, placeholder, disabled = false }) => (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      className="px-3 py-2 bg-slate-600/50 border border-slate-500/50 rounded text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 w-64"
    />
  )

  /**
   * 选择器组件
   */
  const SettingSelect = ({ value, onChange, options }) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="px-3 py-2 bg-slate-600/50 border border-slate-500/50 rounded text-white focus:outline-none focus:border-blue-500 w-32"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )

  /**
   * 标签页导航
   */
  const TabNavigation = () => (
    <div className="flex space-x-1 mb-6 p-1 bg-slate-700/50 rounded-lg w-fit">
      {[
        { id: 'profile', label: '个人信息', icon: User },
        { id: 'account', label: '账户安全', icon: Shield },
        { id: 'preferences', label: '偏好设置', icon: Palette },
        { id: 'system', label: '系统配置', icon: Server },
        { id: 'notifications', label: '通知设置', icon: Bell },
        { id: 'storage', label: '存储管理', icon: HardDrive }
      ].map((tab) => {
        const Icon = tab.icon
        return (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-500 text-white'
                : 'text-slate-300 hover:bg-slate-600/50'
            }`}
          >
            <Icon size={16} />
            <span>{tab.label}</span>
          </button>
        )
      })}
    </div>
  )

  /**
   * 个人信息标签页
   */
  const ProfileTab = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <User size={20} />
          基本信息
        </h3>
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-2xl font-bold">
              {userData.fullName.charAt(0)}
            </div>
            <div className="flex-1">
              <h4 className="text-white font-medium text-lg">{userData.fullName}</h4>
              <p className="text-slate-400">@{userData.username}</p>
              <div className="flex items-center gap-2 mt-2">
                <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 text-xs rounded-full">
                  {userData.role}
                </span>
                <span className="text-slate-400 text-sm">加入于 {userData.joinDate}</span>
              </div>
            </div>
            <button className="px-4 py-2 bg-slate-600/50 text-slate-300 rounded-lg hover:bg-slate-600/70 transition-colors flex items-center gap-2">
              <Edit2 size={16} />
              编辑资料
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-slate-700">
            <SettingInput
              type="text"
              value={userData.fullName}
              placeholder="姓名"
            />
            <SettingInput
              type="text"
              value={userData.username}
              placeholder="用户名"
              disabled
            />
            <SettingInput
              type="email"
              value={userData.email}
              placeholder="邮箱"
            />
            <SettingInput
              type="text"
              value={userData.role}
              placeholder="角色"
              disabled
            />
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Monitor size={20} />
          活动状态
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-slate-700/30 rounded-lg">
            <div className="text-sm text-slate-400 mb-1">最后登录</div>
            <div className="text-white font-medium">{userData.lastLogin}</div>
          </div>
          <div className="p-4 bg-slate-700/30 rounded-lg">
            <div className="text-sm text-slate-400 mb-1">账户状态</div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
              <span className="text-emerald-400 font-medium">活跃</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  /**
   * 账户安全标签页
   */
  const AccountTab = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Lock size={20} />
          密码设置
        </h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">当前密码</label>
            <div className="relative">
              <SettingInput
                type={showPassword ? 'text' : 'password'}
                value="current_password"
                placeholder="输入当前密码"
              />
              <button
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-300"
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-2">新密码</label>
              <SettingInput
                type="password"
                value="new_password"
                placeholder="输入新密码"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-2">确认新密码</label>
              <SettingInput
                type="password"
                value="confirm_password"
                placeholder="再次输入新密码"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Key size={20} />
          API 密钥
        </h3>
        <div className="space-y-4">
          <SettingItem
            icon={Key}
            title="主 API 密钥"
            description="用于 API 访问的主要密钥"
            action={
              <button className="flex items-center gap-1">
                <RefreshCw size={14} />
                重新生成
              </button>
            }
          >
            <div className="flex items-center gap-2">
              <code className="px-2 py-1 bg-slate-600/50 rounded text-xs text-slate-300">
                sk-••••••••••••••••••••••••••••••••
              </code>
              <button className="text-slate-400 hover:text-slate-300">
                <Copy size={16} />
              </button>
            </div>
          </SettingItem>
          <button className="w-full px-4 py-2 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors flex items-center justify-center gap-2">
            <Plus size={16} />
            创建新的 API 密钥
          </button>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Shield size={20} />
          双因素认证
        </h3>
        <SettingItem
          icon={Smartphone}
          title="启用双因素认证"
          description="增加额外的安全层保护您的账户"
          action={<span className="text-emerald-400">已启用</span>}
        >
          <ToggleSwitch checked={true} />
        </SettingItem>
      </div>
    </div>
  )

  /**
   * 偏好设置标签页
   */
  const PreferencesTab = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Palette size={20} />
          外观设置
        </h3>
        <div className="space-y-4">
          <SettingItem
            icon={theme === 'dark' ? Moon : Sun}
            title="主题模式"
            description="选择您喜欢的界面主题"
          >
            <SettingSelect
              value={theme}
              onChange={setTheme}
              options={[
                { value: 'light', label: '浅色' },
                { value: 'dark', label: '深色' },
                { value: 'auto', label: '自动' }
              ]}
            />
          </SettingItem>
          <SettingItem
            icon={Monitor}
            title="界面语言"
            description="选择系统显示语言"
          >
            <SettingSelect
              value={language}
              onChange={setLanguage}
              options={[
                { value: 'zh-CN', label: '中文简体' },
                { value: 'en-US', label: 'English' },
                { value: 'ja-JP', label: '日本語' }
              ]}
            />
          </SettingItem>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Monitor size={20} />
          显示设置
        </h3>
        <div className="space-y-4">
          <SettingItem
            icon={Monitor}
            title="默认页面"
            description="登录后显示的默认页面"
          >
            <SettingSelect
              value="dashboard"
              options={[
                { value: 'dashboard', label: '仪表盘' },
                { value: 'datasets', label: '数据集' },
                { value: 'models', label: '模型' }
              ]}
            />
          </SettingItem>
          <SettingItem
            icon={Monitor}
            title="每页显示数量"
            description="列表页面默认显示的项目数量"
          >
            <SettingSelect
              value="20"
              options={[
                { value: '10', label: '10' },
                { value: '20', label: '20' },
                { value: '50', label: '50' },
                { value: '100', label: '100' }
              ]}
            />
          </SettingItem>
        </div>
      </div>
    </div>
  )

  /**
   * 系统配置标签页
   */
  const SystemTab = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Server size={20} />
          系统信息
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-400">版本</span>
              <span className="text-white">{systemConfig.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">构建日期</span>
              <span className="text-white">{systemConfig.buildDate}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">环境</span>
              <span className="text-white">{systemConfig.environment}</span>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-400">最大用户数</span>
              <span className="text-white">{systemConfig.maxUsers}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">最大数据集</span>
              <span className="text-white">{systemConfig.maxDatasets}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">最大模型数</span>
              <span className="text-white">{systemConfig.maxModels}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Cpu size={20} />
          硬件配置
        </h3>
        <div className="space-y-4">
          <SettingItem
            icon={Zap}
            title="GPU 加速"
            description="启用 GPU 加速计算"
          >
            <ToggleSwitch checked={systemConfig.gpuEnabled} />
          </SettingItem>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-slate-700/30 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">GPU 数量</div>
              <div className="text-white font-medium">{systemConfig.gpuCount}</div>
            </div>
            <div className="p-4 bg-slate-700/30 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">GPU 类型</div>
              <div className="text-white font-medium">{systemConfig.gpuType}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Info size={20} />
          系统操作
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button className="px-4 py-2 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors flex items-center justify-center gap-2">
            <RefreshCw size={16} />
            重启系统
          </button>
          <button className="px-4 py-2 bg-amber-500/20 border border-amber-500/30 text-amber-400 rounded-lg hover:bg-amber-500/30 transition-colors flex items-center justify-center gap-2">
            <Download size={16} />
            导出配置
          </button>
          <button className="px-4 py-2 bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 rounded-lg hover:bg-emerald-500/30 transition-colors flex items-center justify-center gap-2">
            <Upload size={16} />
            导入配置
          </button>
          <button className="px-4 py-2 bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors flex items-center justify-center gap-2">
            <Trash2 size={16} />
            清理缓存
          </button>
        </div>
      </div>
    </div>
  )

  /**
   * 通知设置标签页
   */
  const NotificationsTab = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Bell size={20} />
          通知偏好
        </h3>
        <div className="space-y-4">
          <SettingItem
            icon={Mail}
            title="邮件通知"
            description="通过邮件接收系统通知"
          >
            <ToggleSwitch
              checked={notifications.email}
              onChange={(checked) => setNotifications({...notifications, email: checked})}
            />
          </SettingItem>
          <SettingItem
            icon={Bell}
            title="推送通知"
            description="在浏览器中接收推送通知"
          >
            <ToggleSwitch
              checked={notifications.push}
              onChange={(checked) => setNotifications({...notifications, push: checked})}
            />
          </SettingItem>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">通知类型</h3>
        <div className="space-y-4">
          <SettingItem
            icon={Activity}
            title="训练完成通知"
            description="模型训练完成后通知"
          >
            <ToggleSwitch
              checked={notifications.training}
              onChange={(checked) => setNotifications({...notifications, training: checked})}
            />
          </SettingItem>
          <SettingItem
            icon={Play}
            title="推理完成通知"
            description="推理任务完成后通知"
          >
            <ToggleSwitch
              checked={notifications.inference}
              onChange={(checked) => setNotifications({...notifications, inference: checked})}
            />
          </SettingItem>
          <SettingItem
            icon={AlertTriangle}
            title="系统警告通知"
            description="系统出现问题时通知"
          >
            <ToggleSwitch
              checked={notifications.system}
              onChange={(checked) => setNotifications({...notifications, system: checked})}
            />
          </SettingItem>
        </div>
      </div>
    </div>
  )

  /**
   * 存储管理标签页
   */
  const StorageTab = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <HardDrive size={20} />
          存储使用情况
        </h3>
        <div className="mb-4">
          <div className="flex justify-between mb-2">
            <span className="text-slate-400">总存储配额</span>
            <span className="text-white font-medium">{systemConfig.storageQuota}</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-4">
            <div
              className="h-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
              style={{ width: '45.8%' }}
            ></div>
          </div>
          <div className="flex justify-between mt-2">
            <span className="text-sm text-slate-400">已使用: {systemConfig.usedStorage}</span>
            <span className="text-sm text-slate-400">可用: 542GB</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
          <h4 className="text-white font-medium mb-3 flex items-center gap-2">
            <Database size={18} />
            数据集存储
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">COCO 数据集</span>
              <span className="text-white">25GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">ImageNet 子集</span>
              <span className="text-white">45GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">自定义数据集</span>
              <span className="text-white">128GB</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
          <h4 className="text-white font-medium mb-3 flex items-center gap-2">
            <Server size={18} />
            模型存储
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">YOLOv8 模型</span>
              <span className="text-white">245MB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">ResNet50 模型</span>
              <span className="text-white">98MB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">训练检查点</span>
              <span className="text-white">18GB</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">存储操作</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="px-4 py-2 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors flex items-center justify-center gap-2">
            <Upload size={16} />
            备份数据
          </button>
          <button className="px-4 py-2 bg-amber-500/20 border border-amber-500/30 text-amber-400 rounded-lg hover:bg-amber-500/30 transition-colors flex items-center justify-center gap-2">
            <Trash2 size={16} />
            清理临时文件
          </button>
          <button className="px-4 py-2 bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 rounded-lg hover:bg-emerald-500/30 transition-colors flex items-center justify-center gap-2">
            <RefreshCw size={16} />
            优化存储
          </button>
        </div>
      </div>
    </div>
  )

  /**
   * 渲染当前标签页内容
   */
  const renderTabContent = () => {
    switch (activeTab) {
      case 'profile':
        return <ProfileTab />
      case 'account':
        return <AccountTab />
      case 'preferences':
        return <PreferencesTab />
      case 'system':
        return <SystemTab />
      case 'notifications':
        return <NotificationsTab />
      case 'storage':
        return <StorageTab />
      default:
        return <ProfileTab />
    }
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">系统设置</h1>
          <p className="text-slate-400">管理您的账户和系统配置</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-4 py-2 bg-slate-700/50 border border-slate-600/50 text-slate-300 rounded-lg hover:bg-slate-700/70 transition-colors flex items-center gap-2">
            <RotateCcw size={16} />
            重置设置
          </button>
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2">
            <Save size={16} />
            保存更改
          </button>
        </div>
      </div>

      {/* 标签页导航 */}
      <TabNavigation />

      {/* 标签页内容 */}
      <div className="min-h-[600px]">
        {renderTabContent()}
      </div>

      {/* 帮助信息 */}
      <div className="bg-slate-800/30 border border-slate-700 rounded-xl p-4">
        <div className="flex items-center gap-3">
          <HelpCircle size={20} className="text-slate-400" />
          <div className="flex-1">
            <p className="text-slate-300 text-sm">
              需要帮助？查看我们的
              <a href="#" className="text-blue-400 hover:text-blue-300 mx-1">使用文档</a>
              或联系
              <a href="#" className="text-blue-400 hover:text-blue-300 mx-1">技术支持</a>
            </p>
          </div>
          <button className="text-slate-400 hover:text-slate-300">
            <ExternalLink size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}

export default Settings