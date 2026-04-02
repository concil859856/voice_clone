"""Run: cd workspace root, then: python -m clone"""

import uvicorn

from clone.app.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run(
        "clone.app.main:app",
        host=s.host,
        port=s.port,
        reload=False,
    )
