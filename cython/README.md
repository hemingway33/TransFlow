## 使用以下命令进行cython编译
### python setup.py build_ext --inplace


## MacOS llvm clang 环境构建脚本

第一步：llvm build
cmake -S llvm -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=‘clang;libc;libclc;clang-tools-extra;lldb;lld;polly;openmp;flang;mlir;pstl;cross-project-tests’  -DCMAKE_INSTALL_PREFIX=‘/usr/local’  -DLLVM_TARGETS_TO_BUILD=‘ARM;X86’ -DLLVM_ENABLE_RUNTIMES=‘compiler-rt;libc;libcxx;libcxxabi;libunwind;openmp’
cmake —build build -j8
cmake —install build —prefix /usr/local

第二步: 全局配置
/etc/profile  export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
 第三步：将libunwind.1.dylib移动到python相应的search path环境下 — 由于Mac的SIP机制，无法轻易自主设置link路径，系统会主动ignore环境变量路径