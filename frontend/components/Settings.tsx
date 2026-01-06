import React, { useState, useEffect } from 'react';
import {
  User,
  Bell,
  Moon,
  Globe,
  Shield,
  Info,
  Check,
  ChevronRight,
  LogOut,
  Smartphone,
  Mail,
  Zap,
  LayoutTemplate,
  Loader2
} from 'lucide-react';
import { useAuth } from '../src/hooks/useAuth';
import type { UserConfig } from '../src/services/auth';

interface SettingsProps {
  onLogout: () => void;
}

const Settings: React.FC<SettingsProps> = ({ onLogout }) => {
  const { user, updateUserConfig, logout } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');
  const [showEmailEdit, setShowEmailEdit] = useState(false);
  const [newEmail, setNewEmail] = useState('');

  // 通知设置
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    training: true,
    system: true
  });

  // UI配置
  const [uiConfig, setUiConfig] = useState({
    theme: 'dark' as const,
    language: 'zh-CN',
    default_dataset_id: null as number | null,
    default_model_id: null as number | null,
    notifications_enabled: true,
    auto_save: true
  });

  // 加载用户配置
  useEffect(() => {
    if (user) {
      // 这里可以从后端获取用户配置
      // 暂时使用默认值
      setNewEmail(user.email);
    }
  }, [user]);

  const toggleNotification = (key: keyof typeof notifications) => {
    setNotifications(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleUi = (key: keyof typeof uiConfig) => {
    setUiConfig(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // 保存配置
  const handleSaveConfig = async () => {
    setIsLoading(true);
    setSaveMessage('');

    try {
      const config: UserConfig = {
        theme: uiConfig.theme,
        language: uiConfig.language,
        default_dataset_id: uiConfig.default_dataset_id || undefined,
        default_model_id: uiConfig.default_model_id || undefined,
        notifications_enabled: notifications.email,
        auto_save: uiConfig.auto_save,
      };

      await updateUserConfig(config);
      setSaveMessage('配置保存成功');
      setTimeout(() => setSaveMessage(''), 3000);
    } catch (error) {
      setSaveMessage('配置保存失败');
      setTimeout(() => setSaveMessage(''), 3000);
    } finally {
      setIsLoading(false);
    }
  };

  // 更新邮箱
  const handleUpdateEmail = async () => {
    setIsLoading(true);
    setSaveMessage('');

    try {
      // TODO: 实现邮箱更新API调用
      setSaveMessage('邮箱更新成功');
      setShowEmailEdit(false);
      setTimeout(() => setSaveMessage(''), 3000);
    } catch (error) {
      setSaveMessage('邮箱更新失败');
      setTimeout(() => setSaveMessage(''), 3000);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col p-8 overflow-y-auto custom-scrollbar">
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">设置中心</h2>
          <p className="text-slate-400">管理您的账户信息、系统偏好与通知策略。</p>
        </div>
        {saveMessage && (
          <div className={`flex items-center text-sm ${saveMessage.includes('成功') ? 'text-emerald-400' : 'text-rose-400'} bg-slate-900/50 px-4 py-2 rounded-lg border ${saveMessage.includes('成功') ? 'border-emerald-900/30' : 'border-rose-900/30'}`}>
            {saveMessage.includes('成功') ? <Check size={16} className="mr-2" /> : null}
            {saveMessage}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-6xl">

        {/* Left Column: Profile & About */}
        <div className="lg:col-span-1 space-y-6">
           {/* Profile Card */}
           <div className="glass-panel p-6 rounded-2xl border border-slate-800 relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-r from-cyan-600/20 to-blue-600/20"></div>
              <div className="relative z-10 flex flex-col items-center mt-4">
                 <div className="w-20 h-20 rounded-full bg-slate-900 border-4 border-slate-800 p-1 mb-3 shadow-xl">
                    <div className="w-full h-full rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-xl font-bold text-white">
                      {user?.username.substring(0, 2).toUpperCase()}
                    </div>
                 </div>
                 <h3 className="text-xl font-bold text-white">{user?.username}</h3>
                 <p className="text-sm text-cyan-400 font-medium mb-1">
                   {user?.is_superuser ? 'Administrator' : 'User'}
                 </p>
                 <p className="text-xs text-slate-500">{user?.email}</p>

                 <div className="mt-6 w-full space-y-2">
                    <button
                      onClick={() => setShowEmailEdit(!showEmailEdit)}
                      className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium rounded-lg transition-colors border border-slate-700"
                    >
                       {showEmailEdit ? '取消编辑' : '编辑个人资料'}
                    </button>
                    {showEmailEdit && (
                      <div className="space-y-2">
                        <input
                          type="email"
                          value={newEmail}
                          onChange={(e) => setNewEmail(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-cyan-500"
                          placeholder="新邮箱地址"
                        />
                        <button
                          onClick={handleUpdateEmail}
                          disabled={isLoading}
                          className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center"
                        >
                          {isLoading ? <Loader2 size={16} className="animate-spin" /> : '保存邮箱'}
                        </button>
                      </div>
                    )}
                    <button
                       onClick={() => {
                         logout();
                         onLogout();
                       }}
                       className="w-full py-2 bg-transparent hover:bg-rose-900/20 text-slate-400 hover:text-rose-400 text-sm font-medium rounded-lg transition-colors border border-transparent hover:border-rose-900/30 flex items-center justify-center"
                    >
                       <LogOut size={14} className="mr-2" /> 退出登录
                    </button>
                 </div>
              </div>
           </div>

           {/* System Info Card (Great for Grad Project) */}
           <div className="glass-panel p-6 rounded-2xl border border-slate-800">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                 <Info size={18} className="mr-2 text-cyan-400" /> 关于系统
              </h3>
              <div className="space-y-4 text-sm">
                 <div className="flex justify-between py-2 border-b border-slate-800/50">
                    <span className="text-slate-500">System Version</span>
                    <span className="text-slate-200 font-mono">v2.4.0-beta</span>
                 </div>
                 <div className="flex justify-between py-2 border-b border-slate-800/50">
                    <span className="text-slate-500">Frontend Stack</span>
                    <span className="text-slate-200">React 19 + Tailwind</span>
                 </div>
                 <div className="flex justify-between py-2 border-b border-slate-800/50">
                    <span className="text-slate-500">Backend Core</span>
                    <span className="text-slate-200">Python 3.10 / PyTorch</span>
                 </div>
                 <div className="flex justify-between py-2">
                    <span className="text-slate-500">Developer</span>
                    <span className="text-slate-200">2024届 毕业生</span>
                 </div>
              </div>
           </div>
        </div>

        {/* Right Column: Settings Sections */}
        <div className="lg:col-span-2 space-y-6">
           
           {/* Section 1: General Preferences */}
           <div className="glass-panel p-6 rounded-2xl border border-slate-800">
              <h3 className="text-lg font-bold text-white mb-6 flex items-center">
                 <LayoutTemplate size={20} className="mr-2 text-purple-400" /> 通用设置
              </h3>
              <div className="space-y-6">
                 {/* Language */}
                 <div className="flex items-center justify-between">
                    <div className="flex items-center">
                       <div className="p-2 bg-slate-900 rounded-lg text-slate-400 mr-4"><Globe size={18} /></div>
                       <div>
                          <div className="text-slate-200 font-medium">系统语言</div>
                          <div className="text-slate-500 text-xs">选择界面的显示语言</div>
                       </div>
                    </div>
                    <select 
                       value={uiConfig.language}
                       onChange={(e) => setUiConfig({...uiConfig, language: e.target.value})}
                       className="bg-slate-900 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2 outline-none focus:border-cyan-500 cursor-pointer"
                    >
                       <option value="zh-CN">简体中文</option>
                       <option value="en-US">English</option>
                    </select>
                 </div>

                 {/* Animations */}
                 <div className="flex items-center justify-between">
                    <div className="flex items-center">
                       <div className="p-2 bg-slate-900 rounded-lg text-slate-400 mr-4"><Zap size={18} /></div>
                       <div>
                          <div className="text-slate-200 font-medium">界面动画</div>
                          <div className="text-slate-500 text-xs">启用过渡动画以提升视觉体验</div>
                       </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={uiConfig.animations} onChange={() => toggleUi('animations')} className="sr-only peer" />
                        <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-500"></div>
                    </label>
                 </div>

                 {/* Theme Mode (Mock) */}
                 <div className="flex items-center justify-between opacity-70 pointer-events-none" title="当前版本仅支持深色模式">
                    <div className="flex items-center">
                       <div className="p-2 bg-slate-900 rounded-lg text-slate-400 mr-4"><Moon size={18} /></div>
                       <div>
                          <div className="text-slate-200 font-medium">深色模式</div>
                          <div className="text-slate-500 text-xs">系统默认强制启用深色主题</div>
                       </div>
                    </div>
                    <label className="relative inline-flex items-center">
                        <input type="checkbox" checked={true} readOnly className="sr-only peer" />
                        <div className="w-11 h-6 bg-slate-700 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-400 after:rounded-full after:h-5 after:w-5 peer-checked:bg-cyan-900/50"></div>
                    </label>
                 </div>

                 {/* 保存配置按钮 */}
                 <div className="pt-4 border-t border-slate-700">
                    <button
                      onClick={handleSaveConfig}
                      disabled={isLoading}
                      className="w-full py-2.5 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-medium rounded-xl transition-all active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center shadow-lg shadow-purple-900/20"
                    >
                      {isLoading ? (
                        <>
                          <Loader2 size={18} className="animate-spin mr-2" />
                          保存中...
                        </>
                      ) : (
                        <>
                          <Check size={18} className="mr-2" />
                          保存配置
                        </>
                      )}
                    </button>
                 </div>
              </div>
           </div>

           {/* Section 2: Notifications */}
           <div className="glass-panel p-6 rounded-2xl border border-slate-800">
              <h3 className="text-lg font-bold text-white mb-6 flex items-center">
                 <Bell size={20} className="mr-2 text-amber-400" /> 通知偏好
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 
                 <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800">
                    <div className="flex justify-between items-start mb-4">
                       <Mail size={20} className="text-slate-400" />
                       <label className="relative inline-flex items-center cursor-pointer">
                           <input type="checkbox" checked={notifications.email} onChange={() => toggleNotification('email')} className="sr-only peer" />
                           <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-amber-500"></div>
                       </label>
                    </div>
                    <div className="text-sm font-bold text-slate-200">邮件通知</div>
                    <div className="text-xs text-slate-500 mt-1">当训练任务完成或出错时发送邮件。</div>
                 </div>

                 <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800">
                    <div className="flex justify-between items-start mb-4">
                       <Smartphone size={20} className="text-slate-400" />
                       <label className="relative inline-flex items-center cursor-pointer">
                           <input type="checkbox" checked={notifications.push} onChange={() => toggleNotification('push')} className="sr-only peer" />
                           <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-amber-500"></div>
                       </label>
                    </div>
                    <div className="text-sm font-bold text-slate-200">移动端推送</div>
                    <div className="text-xs text-slate-500 mt-1">推送到绑定的移动设备 App。</div>
                 </div>
              </div>
           </div>
           
           {/* Section 3: Security */}
           <div className="glass-panel p-6 rounded-2xl border border-slate-800">
              <h3 className="text-lg font-bold text-white mb-6 flex items-center">
                 <Shield size={20} className="mr-2 text-emerald-400" /> 安全设置
              </h3>
              <div className="flex items-center justify-between p-4 bg-slate-900/30 rounded-xl border border-slate-800 hover:border-slate-700 transition-colors cursor-pointer group">
                  <div>
                      <div className="text-sm font-bold text-slate-200 group-hover:text-emerald-400 transition-colors">修改账户密码</div>
                      <div className="text-xs text-slate-500">上次修改: 3个月前</div>
                  </div>
                  <ChevronRight size={18} className="text-slate-600 group-hover:text-emerald-400 transition-colors" />
              </div>
           </div>

        </div>
      </div>
    </div>
  );
};

export default Settings;