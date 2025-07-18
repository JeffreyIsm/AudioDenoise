import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,  // Automatically uses your env
  withCredentials: false, // optional: set to true if you use cookies
});

export default api;