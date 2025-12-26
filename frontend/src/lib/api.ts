import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Handle 401 errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: async (email: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    const response = await api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    return response.data;
  },
  register: async (data: { email: string; password: string; full_name: string; role: string }) => {
    const response = await api.post('/auth/register', data);
    return response.data;
  },
  me: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },
};

// Patients API
export const patientsAPI = {
  list: async (params?: { skip?: number; limit?: number; search?: string }) => {
	    // Use trailing slash to match FastAPI router ("/patients/") and avoid redirects
	    const response = await api.get('/patients/', { params });
    return response.data;
  },
  get: async (id: string) => {
    const response = await api.get(`/patients/${id}`);
    return response.data;
  },
  create: async (data: { patient_code: string; date_of_birth?: string; gender?: string; medical_history?: string[]; notes?: string }) => {
	    // Use trailing slash to match FastAPI router ("/patients/") and avoid redirects
	    const response = await api.post('/patients/', data);
    return response.data;
  },
  update: async (id: string, data: Partial<{ patient_code: string; date_of_birth?: string; gender?: string; medical_history?: string[]; notes?: string }>) => {
    const response = await api.put(`/patients/${id}`, data);
    return response.data;
  },
  delete: async (id: string) => {
    await api.delete(`/patients/${id}`);
  },
};

// Documents API
export const documentsAPI = {
  upload: async (patientId: string, file: File, description?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (description) formData.append('description', description);
    const response = await api.post(`/documents/upload/${patientId}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },
  listByPatient: async (patientId: string) => {
    const response = await api.get(`/documents/patient/${patientId}`);
    return response.data;
  },
  get: async (id: string) => {
    const response = await api.get(`/documents/${id}`);
    return response.data;
  },
  delete: async (id: string) => {
    await api.delete(`/documents/${id}`);
  },
};

// Extraction API
export const extractionAPI = {
  process: async (documentId: string) => {
    const response = await api.post(`/extraction/process/${documentId}`);
    return response.data;
  },
  get: async (id: string) => {
    const response = await api.get(`/extraction/${id}`);
    return response.data;
  },
  validate: async (id: string, validatedFields: Array<{ field_name: string; extracted_value: unknown; validated_value?: unknown }>) => {
    const response = await api.post(`/extraction/${id}/validate`, { validated_fields: validatedFields });
    return response.data;
  },
};

// Prediction API
export const predictionAPI = {
  run: async (extractionId: string, modelType: string = 'breast_cancer') => {
    const response = await api.post('/prediction/run', { extraction_id: extractionId, model_type: modelType });
    return response.data;
  },
  listAnalyses: async (params?: { patient_id?: string; skip?: number; limit?: number }) => {
    const response = await api.get('/prediction/analyses', { params });
    return response.data;
  },
  getAnalysis: async (id: string) => {
    const response = await api.get(`/prediction/analyses/${id}`);
    return response.data;
  },
};

// Admin API
export const adminAPI = {
  // Users
  listUsers: async (params?: { skip?: number; limit?: number; search?: string }) => {
    const response = await api.get('/admin/users', { params });
    return response.data;
  },
  getUser: async (id: string) => {
    const response = await api.get(`/admin/users/${id}`);
    return response.data;
  },
  toggleUserActive: async (id: string) => {
    const response = await api.put(`/admin/users/${id}/toggle-active`);
    return response.data;
  },
  updateUserRole: async (id: string, role: string) => {
    const response = await api.put(`/admin/users/${id}/role?role=${role}`);
    return response.data;
  },
  deleteUser: async (id: string) => {
    const response = await api.delete(`/admin/users/${id}`);
    return response.data;
  },
  // Logs
  listLogs: async (params?: { skip?: number; limit?: number; action?: string; user_id?: string }) => {
    const response = await api.get('/admin/logs', { params });
    return response.data;
  },
  // Stats
  getStats: async () => {
    const response = await api.get('/admin/stats');
    return response.data;
  },
};

// Public API (no authentication required)
export const publicAPI = {
  getStats: async () => {
    const response = await axios.get(`${API_BASE_URL.replace('/api/v1', '')}/api/v1/public/stats`);
    return response.data;
  },
};
