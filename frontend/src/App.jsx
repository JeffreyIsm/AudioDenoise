import ClerkProviderWithRoutes from "./auth/ClerkProviderWithRoutes"
import { Routes, Route } from "react-router-dom"

import Home from "./pages/Home"

function App() {

  return (
    <ClerkProviderWithRoutes>
      <Routes>
        <Route path ='/' element={<Home/>} />
      </Routes>
    </ClerkProviderWithRoutes>
  )
}

export default App
