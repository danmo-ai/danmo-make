import { createRouter, createWebHashHistory } from 'vue-router';
import type { RouteRecordRaw } from 'vue-router';

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/image_create',
  },
  {
    path: '/image_create',
    name: 'image_create',
    component: () => import('@/views/ImageCreateView.vue'),
  },
  {
    path: '/video_create',
    name: 'video_create',
    component: () => import('@/views/VideoCreateView.vue'),
  },
  {
    path: '/avatar_create',
    name: 'avatar_create',
    component: () => import('@/views/AvatarCreateView.vue'),
  },
  {
    path: '/audio_create',
    name: 'audio_create',
    component: () => import('@/views/AudioCreateView.vue'),
  },
  {
    path: '/lora_train',
    name: 'lora_train',
    component: () => import('@/views/LoraTrainView.vue'),
  },
  {
    path: '/models',
    name: 'models',
    component: () => import('@/views/ModelsView.vue'),
  },
  {
    path: '/prompts',
    name: 'prompts',
    component: () => import('@/views/PromptsView.vue'),
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue'),
  },
  {
    path: '/assistant',
    name: 'assistant',
    component: () => import('@/views/ChatView.vue'),
  },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
