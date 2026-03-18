import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const AUTH_COOKIE_NAME = "capucinne_basic_auth";

function unauthorizedResponse() {
  return new NextResponse("Authentication Required", {
    status: 401,
    headers: {
      "WWW-Authenticate": 'Basic realm="Secure Area"',
    },
  });
}

function misconfiguredResponse() {
  return new NextResponse("Basic auth is not configured.", {
    status: 503,
  });
}

function parseBasicAuth(header: string) {
  const [scheme, value] = header.split(" ");

  if (scheme !== "Basic" || !value) {
    return null;
  }

  try {
    const decoded = atob(value);
    const separatorIndex = decoded.indexOf(":");

    if (separatorIndex === -1) {
      return null;
    }

    return {
      user: decoded.slice(0, separatorIndex),
      password: decoded.slice(separatorIndex + 1),
    };
  } catch {
    return null;
  }
}

async function createAuthToken(user: string, password: string) {
  const encoded = new TextEncoder().encode(`${user}:${password}`);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function middleware(request: NextRequest) {
  const configuredUser = process.env.BASIC_AUTH_USER?.trim();
  const configuredPassword = process.env.BASIC_AUTH_PASSWORD?.trim();
  const isProtectedDeployment = process.env.VERCEL === "1" || process.env.NODE_ENV === "production";

  if (!configuredUser || !configuredPassword) {
    if (isProtectedDeployment) {
      return misconfiguredResponse();
    }

    return NextResponse.next();
  }

  const expectedToken = await createAuthToken(configuredUser, configuredPassword);
  const cookieToken = request.cookies.get(AUTH_COOKIE_NAME)?.value;

  if (cookieToken === expectedToken) {
    return NextResponse.next();
  }

  const credentials = parseBasicAuth(request.headers.get("authorization") ?? "");

  if (credentials?.user === configuredUser && credentials.password === configuredPassword) {
    const response = NextResponse.next();
    response.cookies.set({
      name: AUTH_COOKIE_NAME,
      value: expectedToken,
      httpOnly: true,
      sameSite: "lax",
      secure: request.nextUrl.protocol === "https:",
      path: "/",
      maxAge: 60 * 60 * 12,
    });
    return response;
  }

  return unauthorizedResponse();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
