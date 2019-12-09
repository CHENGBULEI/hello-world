class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        queue<int>res;
        vector<int>visit(rooms.size(),0);
        int len = rooms.size();
        res.push(0);
        while(!res.empty()){
            
            //���뵱ǰ����
            int nowRoom = res.front();
            //��ǰ�����Ѿ�������
            visit[nowRoom]=1;
            //Ѱ�ҵ�ǰ����������Щ�����Կ��
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
