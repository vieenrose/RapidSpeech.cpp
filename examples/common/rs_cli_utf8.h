#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rs {
namespace cli {

inline void configure_utf8_console()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

class Utf8Args
{
public:
    Utf8Args(int argc, char **argv)
    {
        configure_utf8_console();

#ifdef _WIN32
        if (load_windows_args()) {
            return;
        }
#endif
        storage_.reserve((size_t)argc);
        argv_.reserve((size_t)argc + 1);
        for (int i = 0; i < argc; ++i) {
            storage_.emplace_back(argv[i] ? argv[i] : "");
        }
        rebuild_argv();
    }

    int argc() const
    {
        return argv_.empty() ? 0 : (int)argv_.size() - 1;
    }

    char **argv()
    {
        return argv_.data();
    }

private:
    std::vector<std::string> storage_;
    std::vector<char *> argv_;

    void rebuild_argv()
    {
        argv_.clear();
        argv_.reserve(storage_.size() + 1);
        for (std::string &arg : storage_) {
            argv_.push_back(arg.empty() ? const_cast<char *>("") : &arg[0]);
        }
        argv_.push_back(nullptr);
    }

#ifdef _WIN32
    static bool wide_to_utf8(const wchar_t *wide, std::string &out)
    {
        if (!wide) {
            out.clear();
            return true;
        }

        int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr,
                                    nullptr);
        if (n <= 0) {
            return false;
        }

        out.assign((size_t)n, '\0');
        int written = WideCharToMultiByte(CP_UTF8, 0, wide, -1, &out[0], n,
                                          nullptr, nullptr);
        if (written != n) {
            return false;
        }
        out.pop_back();
        return true;
    }

    bool load_windows_args()
    {
        using CommandLineToArgvWFn = LPWSTR *(WINAPI *)(LPCWSTR, int *);

        HMODULE shell32 = LoadLibraryW(L"shell32.dll");
        if (!shell32) {
            return false;
        }

        auto *fn = reinterpret_cast<CommandLineToArgvWFn>(
            GetProcAddress(shell32, "CommandLineToArgvW"));
        if (!fn) {
            FreeLibrary(shell32);
            return false;
        }

        int wargc = 0;
        LPWSTR *wargv = fn(GetCommandLineW(), &wargc);
        if (!wargv || wargc <= 0) {
            if (wargv) {
                LocalFree(wargv);
            }
            FreeLibrary(shell32);
            return false;
        }

        storage_.clear();
        storage_.reserve((size_t)wargc);
        for (int i = 0; i < wargc; ++i) {
            std::string arg;
            if (!wide_to_utf8(wargv[i], arg)) {
                LocalFree(wargv);
                FreeLibrary(shell32);
                storage_.clear();
                return false;
            }
            storage_.push_back(std::move(arg));
        }

        LocalFree(wargv);
        FreeLibrary(shell32);
        rebuild_argv();
        return true;
    }
#endif
};

} // namespace cli
} // namespace rs
