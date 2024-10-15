#include "utf8.h"
#include "ACAutomachine.h"

#include <queue>
#include <memory>
#include <vector>
#include <iostream>
#include <unordered_map>
using namespace std;
int main(){
    
    vector<string> parttern ={"1","random","你好"};
    string text ="1112312311123";
    vector<vector<uint32_t>> stCodes;
    vector<uint32_t> stCode;
    CUTF8Handle cutf;
    for(const auto &i:parttern){
        stCode.clear();
        cutf.Decode(i,stCode);
        // cout<<stCode.size()<<'\n';
        stCodes.push_back(stCode);
    }

    CACAutomaton<uint32_t> st;
    st.BuildGoto(stCodes);
    stCode.clear();
    cutf.Decode(text,stCode);
    st.Replace(stCode,'*');
    text = "";
    cutf.Encode(stCode,text);
    cout<<text;
}
