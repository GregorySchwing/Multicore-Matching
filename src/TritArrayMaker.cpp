/*
Copyright 2022, Gregory Schwing.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "TritArrayMaker.h"

cpp_int TritArrayMaker::large_fact(int num) {
cpp_int fact = 1;
for (int i=num; i>1; --i)
    fact *= i;
return fact;
}

cpp_int TritArrayMaker::large_pow(int num) {
if (num == 0){
    cpp_int fact = 1;
    return fact;
}
cpp_int fact = 3;
for (int i=num; i>1; --i)
    fact *= 3;
return fact;
}


// https://stackoverflow.com/questions/12015752/conversion-of-binary-bitstream-to-and-from-ternary-bitstream
// Compute the trit representation of the bits in the given
// byte buffer.  The highest order byte is bytes[0].  The
// lowest order trit in the output is trits[0].  This is 
// not a very efficient algorithm, but it doesn't use any
// division.  If the output buffer is too small, high order
// trits are lost.
void TritArrayMaker::to_trits(Byte *bytes, int n_bytes, 
            Byte *trits, int n_trits)
{
    int i_trit, i_byte, mask;

    for (i_trit = 0; i_trit < n_trits; i_trit++)
        trits[i_trit] = 0;

    // Scan bits left to right.
    for (i_byte = 0; i_byte < n_bytes; i_byte++) {

        Byte byte = bytes[i_byte];

        for (mask = 0x80; mask; mask >>= 1) {
        // Compute the next bit.
        int bit = (byte & mask) != 0;

        // Update the trit representation
        trits[0] = trits[0] * 2 + bit;
        for (i_trit = 1; i_trit < n_trits; i_trit++) {
            trits[i_trit] *= 2;
            if (trits[i_trit - 1] > 2) {
            trits[i_trit - 1] -= 3;
            trits[i_trit]++;
            }
        }
        }
    }
}

std::vector<Byte> TritArrayMaker::create_trits(cpp_int x){
    // export into 8-bit unsigned values, most significant bit first:
    std::vector<Byte> bytes;
    std::vector<Byte> trits;
    //std::cout << "trits input " << x << std::endl;
    export_bits(x, std::back_inserter(bytes), 8);

    int N_BYTES = bytes.size();
    // For some reason, N_BYTES is 1 less than the right number on occasion
    int N_TRITS = (N_BYTES*8*2)/3 + 1;
    printf("N_BYTES %d\n", N_BYTES);
    printf("N_TRITS %d\n", N_TRITS);


    // Make a trit buffer.
    //Byte trits [N_TRITS];
    trits.resize(N_TRITS);
    to_trits(bytes.data(), N_BYTES, trits.data(), N_TRITS);
    for (int j = N_TRITS - 1; j >= 0; j--) {
        printf("%d", trits[j]);
    }
    return trits;
}
