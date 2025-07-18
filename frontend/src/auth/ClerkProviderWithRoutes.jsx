import { ClerkProvider } from "@clerk/clerk-react";

import { HashRouter } from "react-router-dom"
{/* BrowserRouter instead of HashRouter for real, this is just for github */}

// Import your Publishable Key
const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

if (!PUBLISHABLE_KEY) {
  throw new Error('Missing Publishable Key')
}

export default function ClerkProviderWithRoutes({children}){
    return(
        <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
            <HashRouter>{children}</HashRouter>
        </ClerkProvider>
    )
}