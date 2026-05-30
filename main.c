#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "api.h"

int main(void)
{
    uint8_t pk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_SECRETKEYBYTES];
    uint8_t ct[PQCLEAN_MLKEM768_CLEAN_CRYPTO_CIPHERTEXTBYTES];
    uint8_t ss_enc[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
    uint8_t ss_dec[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];

    int ok = 0;
    for (int i = 0; i < 100000; i++) {
        // Key pair generation
        if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(pk, sk) != 0) {
            printf("Key pair generation failed\n");
            continue;
        }

        // Encapsulation
        if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(ct, ss_enc, pk) != 0) {
            printf("Encapsulation failed\n");
            continue;
        }

        // Decapsulation
        if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(ss_dec, ct, sk) != 0) {
            printf("Decapsulation failed\n");
            continue;
        }

        // Verify that the shared secrets match
        if (memcmp(ss_enc, ss_dec, PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES) == 0) {
            // printf("Shared secrets match. Test passed.\n");
            ok++;
        } else {
            printf("Shared secrets do not match. Test failed.\n");
        }
        if (i % 10000 == 0) {
            printf("Progress: %d\n", i);
        }
    }
    printf("Test results: %d/%d passed.\n", ok, 10000);     
}