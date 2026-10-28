/* nn::diag: the game's own log lines and aborts. */
#include "hos.h"
#include <stdio.h>

/* nn::diag::detail::PutImpl(LogMetaData const&, char const* text, unsigned long length) */
HLE(_ZN2nn4diag6detail7PutImplERKNS0_11LogMetaDataEPKcm) {
    uint64_t n = ARG(2) > 4000 ? 4000 : ARG(2);
    char buf[4001];
    memcpy(buf, GP(ARG(1)), n);
    buf[n] = 0;
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    hos_log("game: %s", buf);
}

/* nn::diag::detail::AbortImpl() */
HLE(_ZN2nn4diag6detail9AbortImplEv) { hos_fatal(c, "the game aborted (nn::diag::detail::AbortImpl)"); }

/* nn::detail::UnexpectedDefaultImpl(): a switch reached a case the SDK says cannot happen */
HLE(_ZN2nn6detail21UnexpectedDefaultImplEv) { hos_fatal(c, "nn::detail::UnexpectedDefaultImpl"); }

/* nn::diag::GetBacktrace(unsigned long*, int) -> int: none */
HLE(_ZN2nn4diag12GetBacktraceEPmi) { RET(0); }
