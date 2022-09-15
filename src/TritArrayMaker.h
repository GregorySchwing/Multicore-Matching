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
#ifndef TRIT_ARRAY_MAKER_H
#define TRIT_ARRAY_MAKER_H

#include<iostream>
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
//using namespace std;
typedef unsigned char Byte;

class TritArrayMaker
{
    public:
        static std::vector<Byte> create_trits(cpp_int x);
        static cpp_int large_pow(int num);

    private:
        static cpp_int large_fact(int num);
        static void to_trits(Byte *bytes, int n_bytes, 
                Byte *trits, int n_trits);
};

#endif