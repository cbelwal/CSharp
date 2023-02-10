using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public class CNode
    {
        public int idx;
        public double value;
        public int level;
        public CNode [] children;
        public CNode parent;
    }
    
    public class CCombinationGenerator
    {
        List<List<double>> _masterList;
        
        
        /// <summary>
        /// Generate all possible subsets of length 'iSizeOfSubset' of elements 
        /// that are in the set mainSet. NOTE: Ordering of sets do not matter
        /// Eg. if input if 0,1,2 and sizeOfSubset = 2 then
        /// Call Gensets each tome to generate 0,1 and 0,2
        /// </summary>
        /// <param name="mainSet"></param>
        /// <param name="iSizeOfSubset"></param>
        /// <returns></returns>
        public List<List<double>> GenerateCombinations(List<double> mainSet, int iSizeOfSubset)
        {
            _masterList = new List<List<double>>();
            CNode root = CreateTree(mainSet, iSizeOfSubset);
            GenerateCombinationsFromTree(root, iSizeOfSubset);
            //int ii;
            //for (ii = 0; ii + iSizeOfSubset <= mainSet.Count; ii++)
            //    GenSets(mainSet, ii,iSizeOfSubset);
            

            return _masterList;
        }

        /// <summary>
        /// Returns the outpur of (n k)
        /// </summary>
        /// <param name="lElementsInSet"></param>
        /// <param name="lElementsInSubset"></param>
        /// <returns></returns>
        public long GetNumberOfCombinations(long lElementsInSet, long lElementsInSubset)
        {
            long lStopAt;
            long comb;

            if (lElementsInSet < lElementsInSubset) return 0;

            lStopAt = lElementsInSet - lElementsInSubset;

            if (lStopAt < lElementsInSubset)
            {
                lStopAt = lElementsInSubset;
                comb = CMath.Factorial(lElementsInSet, lStopAt) / CMath.Factorial(lElementsInSet - lElementsInSubset,1);
            }
            else
            {
                comb = CMath.Factorial(lElementsInSet, lStopAt) / CMath.Factorial(lElementsInSubset,1);
            }

            return comb;

        }

        /// <summary>
        /// Private routine to generate task sets
        /// Generate sets of type 0,1 and 0,2. The startIdx remains fixed
        /// </summary>
        /// <param name="mainSet"></param>
        /// <param name="iStartIdx"></param>
        /// <param name="iSizeOfSubset"></param>
        //private void GenSets(List<double> mainSet, int iStartIdx, int iSizeOfSubset)
        //{
        //    int ii,jj;
        //    double dInitialValue;
        //    List<double> newList; 

        //    dInitialValue = mainSet[iStartIdx];
        //    for (ii = iStartIdx+1; ii + iSizeOfSubset - 1 <= mainSet.Count; ii++)
        //    {
        //        newList =  new List<double>();
        //        for (jj = ii; jj < ii + iSizeOfSubset-1; jj++)
        //            newList.Add(mainSet[jj]);

        //        newList.Add(dInitialValue);

        //        _masterList.Add(newList);
        //    }
        //}

        /// <summary>
        /// Adds Child node uptill max
        /// </summary>
        /// <param name="node"></param>
        /// <param name="value"></param>
        /// <param name="max"></param>
        private void AddChildren(List<double> mainSet, CNode node, int iSizeOfSubset)
        {
            int ii,noOfChildren;
            CNode newNode;
            int idx;

            if (node.level == iSizeOfSubset) return;

            if (node.idx < 0) noOfChildren = mainSet.Count;
            else noOfChildren = mainSet.Count - (node.idx + 1);
            
            node.children = new CNode[noOfChildren];
            idx = 0;
            for(ii=node.idx + 1;ii < mainSet.Count ;ii++)
            {
                node.children[idx] = new CNode();
                newNode = node.children[idx++];
                newNode.value = mainSet[ii];
                newNode.idx = ii;
                newNode.parent = node;
                newNode.level = node.level + 1;
                AddChildren(mainSet, newNode, iSizeOfSubset);
            }

        }

        /// <summary>
        /// Create the tree with the values
        /// </summary>
        /// <param name="mainSet"></param>
        /// <param name="iSizeOfSubset"></param>
        private CNode CreateTree(List<double> mainSet, int iSizeOfSubset)
        {
            CNode root = new CNode();
            root.value = -1;
            root.idx = -1;
            root.level = 0;
            AddChildren(mainSet, root, iSizeOfSubset);
            return root;
        }

        /// <summary>
        /// Generate Combinations using Tree Transversal
        /// </summary>
        /// <param name="node"></param>
        /// <param name="iSizeOfSubset"></param>
        private void GenerateCombinationsFromTree(CNode node, int iSizeOfSubset)
        {
            if (node.children == null)
            {
                if (node.level == iSizeOfSubset) GenerateCombination(node);
                return;
            }
            
            foreach (CNode child in node.children)
            {
                GenerateCombinationsFromTree(child, iSizeOfSubset);
            }
        }

        /// <summary>
        /// Generates a Combination using leaf node and stores it in _masterList
        /// Only send leaf nodes upto a certain level
        /// </summary>
        /// <param name="leafNode"></param>
        private void GenerateCombination(CNode leafNode)
        {
            List<double> subSet = new List<double>();
            CNode node = leafNode;
            
            while (node.idx >= 0)
            {
                subSet.Add(node.value);
                node = node.parent;
            }

            _masterList.Add(subSet);
        }

    }
}
