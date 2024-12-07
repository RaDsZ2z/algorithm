#pragma once

#include <queue>
#include <memory>
#include <vector>
#include <iostream>
#include <unordered_map>

template <typename T>
class CACAutomaton
{
    typedef typename std::unordered_map<T, size_t> next_state_hash_type;
    typedef std::shared_ptr<next_state_hash_type>  next_state_hash_ptr_type;

    struct state_node_type
    {
        bool                        m_bEnd;
        size_t                      m_uiLevel;
        size_t                      m_uiFailIndex;
        size_t                      m_uiParentIndex;
        next_state_hash_ptr_type    m_stNextHashPtr;//指向map的指针
    };

    typedef std::vector<state_node_type> trie_tree_type;

public:
    template <typename U>
    void BuildACFSM(const U& stPatterns)
    {
		m_stTrieTree.clear();

        BuildGoto(stPatterns);

        BuildFail();
    }

    template <typename U>
    void Replace(U& stTarget, const T& stReplace)
    {
        if (m_stTrieTree.size() < 2)
            return;

        size_t uiIndex = 0;
        auto stIter = stTarget.begin();
        while (stIter != stTarget.end())
        {
            if (m_stTrieTree[uiIndex].m_stNextHashPtr.get())
            {
                auto stTrieIter = m_stTrieTree[uiIndex].m_stNextHashPtr->find(*stIter);
                if (stTrieIter != m_stTrieTree[uiIndex].m_stNextHashPtr->end())
                {
                    uiIndex = stTrieIter->second;
                }
                else
                {
                    if (!uiIndex) ++stIter;
                    uiIndex = m_stTrieTree[uiIndex].m_uiFailIndex;
                    continue;
                }
            }
            else
            {
                if (!uiIndex) ++stIter;
                uiIndex = m_stTrieTree[uiIndex].m_uiFailIndex;
                continue;
            }

            if (m_stTrieTree[uiIndex].m_bEnd)
            {
                for (size_t offset = 0; offset < m_stTrieTree[uiIndex].m_uiLevel; ++offset)
                {
                    *(stIter - offset) = stReplace;
                }
            }
            ++stIter;
        }
    }

    template <typename U>
    bool Find(U& stTarget)
    {
        if (m_stTrieTree.size() < 2)
            return false;

        size_t uiIndex = 0;
        auto stIter = stTarget.begin();
        while (stIter != stTarget.end())
        {
            if (m_stTrieTree[uiIndex].m_stNextHashPtr.get())
            {
                auto stTrieIter = m_stTrieTree[uiIndex].m_stNextHashPtr->find(*stIter);
                if (stTrieIter != m_stTrieTree[uiIndex].m_stNextHashPtr->end())
                {
                    uiIndex = stTrieIter->second;
                }
                else
                {
                    if (!uiIndex) ++stIter;
                    uiIndex = m_stTrieTree[uiIndex].m_uiFailIndex;
                    continue;
                }
            }
            else
            {
                if (!uiIndex) ++stIter;
                uiIndex = m_stTrieTree[uiIndex].m_uiFailIndex;
                continue;
            }

            if (m_stTrieTree[uiIndex].m_bEnd) return true;

            ++stIter;
        }

        return false;
    }

    void Dump()
    {
        typename std::pair<T, size_t> stIndex;
        typename std::queue<std::pair<T, size_t>> stQueue;
        stQueue.push(std::make_pair('0', 0));

        while (!stQueue.empty())
        {
            stIndex = stQueue.front();
            stQueue.pop();

            std::cout << (T)(stIndex.first) << "|" << stIndex.second << "|"
                      << m_stTrieTree[stIndex.second].m_bEnd << "|"
                      << m_stTrieTree[stIndex.second].m_uiLevel << "|"
                      << m_stTrieTree[stIndex.second].m_uiFailIndex << "|"
                      << m_stTrieTree[stIndex.second].m_uiParentIndex << std::endl;

            auto pstNextHash = m_stTrieTree[stIndex.second].m_stNextHashPtr.get();
            if (pstNextHash != nullptr)
            {
                for (auto& stElem : *pstNextHash)
                {
                    stQueue.push(std::make_pair(stElem.first, stElem.second));
                }
            }
        }
    }

// protected:
    template <typename U>
    void BuildGoto(const U& stPatterns)
    {
        // std::cout<<"stPatterns.size():"<<stPatterns.size()<<'\n';
        state_node_type stState;
        stState.m_bEnd = false;
        stState.m_uiLevel = 0;
        stState.m_uiFailIndex = 0;
        stState.m_uiParentIndex = 0;
        m_stTrieTree.push_back(stState);

        for (const auto& stPattern : stPatterns)
        {
            size_t uiStateIndex = 0;
            for (const auto& stElem : stPattern)
            {
                if (m_stTrieTree[uiStateIndex].m_stNextHashPtr.get() == nullptr)
                {
                    // 没有任何子节点
                    m_stTrieTree[uiStateIndex].m_stNextHashPtr = std::make_shared<next_state_hash_type>();
                }
                else
                {
                    //有子节点
                    auto stNextIter = m_stTrieTree[uiStateIndex].m_stNextHashPtr->find(stElem);
                    if (stNextIter != m_stTrieTree[uiStateIndex].m_stNextHashPtr->end())
                    {
                        //有当前字符对应的子节点
                        uiStateIndex = stNextIter->second;
                        continue;
                    }
                }
                //创建了新的子节点
                stState.m_uiParentIndex = uiStateIndex;
                stState.m_uiLevel = m_stTrieTree[uiStateIndex].m_uiLevel + 1;
                m_stTrieTree.push_back(stState);
                (*(m_stTrieTree[uiStateIndex].m_stNextHashPtr))[stElem] = m_stTrieTree.size() - 1; 
                uiStateIndex = m_stTrieTree.size() - 1;
            }
            m_stTrieTree[uiStateIndex].m_bEnd = true;
        }
        // std::cout<<"m_stTrieTree.size():"<<m_stTrieTree.size()<<'\n';
    }

    void BuildFail()
    {
        size_t uiFailIndex;
        typename std::pair<T, size_t> stIndex;
        typename std::queue<std::pair<T, size_t>> stQueue;
        m_stTrieTree[0].m_uiFailIndex = 0;

        for (const auto& stElem : *(m_stTrieTree[0].m_stNextHashPtr))
        {
            m_stTrieTree[stElem.second].m_uiFailIndex = 0;
            stQueue.push(std::make_pair(stElem.first, stElem.second));
        }

        while (!stQueue.empty())
        {
            stIndex = stQueue.front();
            stQueue.pop();

            auto pstNextHash = m_stTrieTree[stIndex.second].m_stNextHashPtr.get();
            if (pstNextHash != nullptr)
            {
                for (const auto& stElem : *pstNextHash)
                {
                    stQueue.push(std::make_pair(stElem.first, stElem.second));
                }
            }

            if (m_stTrieTree[stIndex.second].m_uiParentIndex == 0)
                continue;

            uiFailIndex = m_stTrieTree[m_stTrieTree[stIndex.second].m_uiParentIndex].m_uiFailIndex;
            while (true)
            {
                if (m_stTrieTree[uiFailIndex].m_stNextHashPtr.get())
                {
                    auto stIter = m_stTrieTree[uiFailIndex].m_stNextHashPtr->find(stIndex.first);
                    if (stIter != m_stTrieTree[uiFailIndex].m_stNextHashPtr->end())
                    {
                        m_stTrieTree[stIndex.second].m_uiFailIndex = stIter->second;
                        break;
                    }

                    if (!uiFailIndex) break;
                }

                uiFailIndex = m_stTrieTree[uiFailIndex].m_uiFailIndex;
            }
        }
    }

private:
    trie_tree_type   m_stTrieTree;
};
