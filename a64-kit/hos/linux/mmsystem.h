#pragma once
#include <windows.h>
#define TIMERR_NOERROR 0
static inline unsigned timeBeginPeriod(unsigned p) { (void)p; return 0; }
static inline unsigned timeEndPeriod(unsigned p) { (void)p; return 0; }
DWORD timeGetTime(void);

/* waveOut on the web: the frames go to a ring in memory (win32.c) that the page's AudioWorklet plays */
typedef void *HWAVEOUT;
typedef struct { WORD wFormatTag, nChannels; DWORD nSamplesPerSec, nAvgBytesPerSec; WORD nBlockAlign, wBitsPerSample, cbSize; } WAVEFORMATEX;
typedef struct { LPSTR lpData; DWORD dwBufferLength, dwBytesRecorded; DWORD_PTR dwUser; DWORD dwFlags, dwLoops; void *lpNext; DWORD_PTR reserved; } WAVEHDR;
typedef UINT MMRESULT;
#define WAVE_FORMAT_PCM 1
#define WAVE_MAPPER ((UINT)-1)
#define CALLBACK_NULL 0
#define MMSYSERR_NOERROR 0
#define MMSYSERR_NODRIVER 6
#define WHDR_DONE 1
#define WHDR_PREPARED 2
MMRESULT waveOutOpen(HWAVEOUT *w, UINT d, const WAVEFORMATEX *f, DWORD_PTR cb, DWORD_PTR u, DWORD fl);
MMRESULT waveOutWrite(HWAVEOUT w, WAVEHDR *h, UINT n);
static inline MMRESULT waveOutPrepareHeader(HWAVEOUT w, WAVEHDR *h, UINT n) {
    (void)w; (void)n;
    if (h) h->dwFlags |= WHDR_PREPARED;
    return 0;
}
static inline MMRESULT waveOutUnprepareHeader(HWAVEOUT w, WAVEHDR *h, UINT n) {
    (void)w; (void)n;
    if (h) h->dwFlags &= ~(DWORD)WHDR_PREPARED;
    return 0;
}
static inline MMRESULT waveOutReset(HWAVEOUT w) { (void)w; return 0; }
static inline MMRESULT waveOutClose(HWAVEOUT w) { (void)w; return 0; }
