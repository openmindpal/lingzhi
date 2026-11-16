/* 桥接器 ↔ 包装库 跨语言对拍探针——C 消费端直链 libmindpal_dds_std（引擎 dlopen 同款库）。
 * 用途：仿真验证包 compose 的 ddsinterop 门——Python 桥接器发布的 mindpal::JointStates
 * 必须被引擎同款 C 包装库全块读回（bits=0x1F），线契约逐字段可解析。
 * 用法: std_probe <domain> <topic>
 * 退出码：0 = PROBE_OK（全块快照 + 四元数归一 + q 全有限）；1 = 超时/失配。
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>

int32_t mp_dds_init(int32_t domain);
int32_t mp_dds_read_lowstate_full(const char* topic, float* q, float* dq, float* tau,
                                  float* imu, float* ft, int32_t ft_capacity,
                                  int32_t* out_n_ft, int32_t n);

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "用法: std_probe <domain> <topic>\n"); return 2; }
    int domain = atoi(argv[1]);
    const char* topic = argv[2];
    float q[35], dq[35], tau[35], imu[10], ft[48];
    int32_t n_ft = 0;
    printf("init: %d\n", mp_dds_init(domain));
    for (int round = 0; round < 50; round++) {
        int32_t bits = mp_dds_read_lowstate_full(topic, q, dq, tau, imu, ft, 8, &n_ft, 35);
        if (bits & 1) {
            float qn = sqrtf(imu[0]*imu[0] + imu[1]*imu[1] + imu[2]*imu[2] + imu[3]*imu[3]);
            int finite_ok = 1;
            for (int i = 0; i < 35; i++) if (!isfinite(q[i])) { finite_ok = 0; break; }
            int ok = (bits & 0x3) == 0x3 && finite_ok && qn > 0.9f && qn < 1.1f;
            printf("round %d: bits=0x%x q=%g,%g,%g dq=%g,%g tau=%g,%g imu=%g,%g,%g,%g n_ft=%d\n",
                   round, bits, q[0], q[3], q[34], dq[0], dq[1], tau[0], tau[1],
                   imu[0], imu[1], imu[8], imu[9], n_ft);
            {   /* 边界原始位转储——tau 尾/imu 头/n_ft/ft 头（线字节对位诊断） */
                const uint32_t *tau_u = (const uint32_t *)tau;
                const uint32_t *imu_u = (const uint32_t *)imu;
                const uint32_t *ft_u  = (const uint32_t *)ft;
                printf("  raw: tau[32..34]=%08x %08x %08x imu[0..3]=%08x %08x %08x %08x n_ft=%08x ft00=%08x\n",
                       tau_u[32], tau_u[33], tau_u[34],
                       imu_u[0], imu_u[1], imu_u[2], imu_u[3], n_ft, ft_u[0]);
            }
            printf(ok ? "PROBE_OK\n" : "PROBE_MISMATCH\n");
            return ok ? 0 : 1;
        }
        usleep(100000);
    }
    printf("PROBE_TIMEOUT\n");
    return 1;
}
