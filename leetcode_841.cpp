class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        queue<int>res;
        vector<int>visit(rooms.size(),0);
        int len = rooms.size();
        res.push(0);
        while(!res.empty()){
            
            //进入当前房间
            int nowRoom = res.front();
            //当前房间已经被访问
            visit[nowRoom]=1;
            //寻找当前房间里有哪些房间的钥匙
            int keyLen = rooms[nowRoom].size();
            for(int i=0;i<keyLen;i++){
                if(visit[rooms[nowRoom][i]]==0){
                    res.push(rooms[nowRoom][i]);
                }
            }
            res.pop();
        }
        for(int i=0;i<len;i++){
            if(visit[i]==0)return false;
        }
        return true;
    }
};
