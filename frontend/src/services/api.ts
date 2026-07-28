
import axios from 'axios'
import type {
  TripFormData,
  TripPlanResponse,
  TripReviewRequest,
  TripStreamEvent,
  TripStreamEventName
} from '@/types'


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 多Agent长任务最多等待5分钟
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

/**
 * 通过POST SSE按天接收旅行计划。
 */
export async function generateTripPlanStream(
  formData: TripFormData,
  onEvent: (event: TripStreamEvent) => void
): Promise<TripPlanResponse> {
  const response = await fetch(
    `${API_BASE_URL.replace(/\/$/, '')}/api/trip/plan/stream`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream'
      },
      body: JSON.stringify(formData)
    }
  )

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `流式请求失败：HTTP ${response.status}`)
  }
  if (!response.body) {
    throw new Error('浏览器未提供流式响应读取能力')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalResponse: TripPlanResponse | null = null

  const consumeBlock = (block: string) => {
    let eventName: TripStreamEventName = 'status'
    const dataLines: string[] = []

    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim() as TripStreamEventName
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim())
      }
    }
    if (!dataLines.length) return

    const payload = JSON.parse(dataLines.join('\n'))
    const streamEvent: TripStreamEvent = {
      event: eventName,
      ...payload
    }
    onEvent(streamEvent)

    if (eventName === 'review' || eventName === 'complete' || eventName === 'error') {
      finalResponse = payload as TripPlanResponse
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      consumeBlock(block)
      boundary = buffer.indexOf('\n\n')
    }
  }

  buffer += decoder.decode()
  if (buffer.trim()) {
    consumeBlock(buffer.trim())
  }
  if (!finalResponse) {
    throw new Error('流式连接已结束，但没有收到最终旅行计划状态')
  }
  return finalResponse
}

/**
 * 提交人工审核结果并恢复旅行规划工作流
 */
export async function resumeTripPlan(
  threadId: string,
  review: TripReviewRequest
): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>(
      `/api/trip/plan/${encodeURIComponent(threadId)}/resume`,
      review
    )
    return response.data
  } catch (error: any) {
    console.error('恢复旅行规划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '恢复旅行规划失败')
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient
