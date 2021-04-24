use std::mem;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign};

use retain_mut::RetainMut;

use crate::bitmap::container::Container;
use crate::RoaringBitmap;

impl RoaringBitmap {
    /// Unions in-place with the specified other bitmap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..5).collect();
    /// let rb3: RoaringBitmap = (1..5).collect();
    ///
    /// rb1.union_with(&rb2);
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    ///
    /// Can also be done via the `BitOr` operator.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..5).collect();
    /// let rb3: RoaringBitmap = (1..5).collect();
    ///
    /// let rb1 = rb1 | rb2;
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    #[deprecated(
        since = "0.6.7",
        note = "Please use the `BitOrAssign::bitor_assign` ops method instead"
    )]
    pub fn union_with(&mut self, other: &RoaringBitmap) {
        BitOrAssign::bitor_assign(self, other)
    }

    /// Intersects in-place with the specified other bitmap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..5).collect();
    /// let rb3: RoaringBitmap = (3..4).collect();
    ///
    /// rb1.intersect_with(&rb2);
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    ///
    /// Can also be done via the `BitAnd` operator.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..5).collect();
    /// let rb3: RoaringBitmap = (3..4).collect();
    ///
    /// let rb1 = rb1 & rb2;
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    #[deprecated(
        since = "0.6.7",
        note = "Please use the `BitAndAssign::bitand_assign` ops method instead"
    )]
    pub fn intersect_with(&mut self, other: &RoaringBitmap) {
        self.containers.retain_mut(|cont| {
            match other.containers.binary_search_by_key(&cont.key, |c| c.key) {
                Ok(loc) => {
                    cont.intersect_with(&other.containers[loc]);
                    cont.len != 0
                }
                Err(_) => false,
            }
        })
    }

    /// Removes all values in the specified other bitmap from self, in-place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..5).collect();
    /// let rb3: RoaringBitmap = (1..3).collect();
    ///
    /// rb1.difference_with(&rb2);
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    ///
    /// Can also be done via the `Sub` operator.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..5).collect();
    /// let rb3: RoaringBitmap = (1..3).collect();
    ///
    /// let rb1 = rb1 - rb2;
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    #[deprecated(
        since = "0.6.7",
        note = "Please use the `SubAssign::sub_assign` ops method instead"
    )]
    pub fn difference_with(&mut self, other: &RoaringBitmap) {
        self.containers.retain_mut(|cont| {
            match other.containers.binary_search_by_key(&cont.key, |c| c.key) {
                Ok(loc) => {
                    cont.difference_with(&other.containers[loc]);
                    cont.len != 0
                }
                Err(_) => true,
            }
        })
    }

    /// Replaces this bitmap with one that is equivalent to `self XOR other`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..6).collect();
    /// let rb3: RoaringBitmap = (1..3).chain(4..6).collect();
    ///
    /// rb1.symmetric_difference_with(&rb2);
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    ///
    /// Can also be done via the `BitXor` operator.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb1: RoaringBitmap = (1..4).collect();
    /// let rb2: RoaringBitmap = (3..6).collect();
    /// let rb3: RoaringBitmap = (1..3).chain(4..6).collect();
    ///
    /// let rb1 = rb1 ^ rb2;
    ///
    /// assert_eq!(rb1, rb3);
    /// ```
    #[deprecated(
        since = "0.6.7",
        note = "Please use the `BitXorAssign::bitxor_assign` ops method instead"
    )]
    pub fn symmetric_difference_with(&mut self, other: &RoaringBitmap) {
        for container in &other.containers {
            let key = container.key;
            match self.containers.binary_search_by_key(&key, |c| c.key) {
                Err(loc) => self.containers.insert(loc, container.clone()),
                Ok(loc) => {
                    self.containers[loc].symmetric_difference_with(container);
                    if self.containers[loc].len == 0 {
                        self.containers.remove(loc);
                    }
                }
            }
        }
    }
}

impl BitOr<RoaringBitmap> for RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `union` between both maps.
    fn bitor(mut self, rhs: RoaringBitmap) -> RoaringBitmap {
        BitOrAssign::bitor_assign(&mut self, rhs);
        self
    }
}

impl BitOr<&RoaringBitmap> for RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `union` between both maps.
    fn bitor(mut self, rhs: &RoaringBitmap) -> RoaringBitmap {
        BitOrAssign::bitor_assign(&mut self, rhs);
        self
    }
}

impl BitOr<RoaringBitmap> for &RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `union` between both maps.
    fn bitor(self, rhs: RoaringBitmap) -> RoaringBitmap {
        BitOr::bitor(rhs, self)
    }
}

impl BitOr<&RoaringBitmap> for &RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `union` between both maps.
    fn bitor(self, rhs: &RoaringBitmap) -> RoaringBitmap {
        if self.len() <= rhs.len() {
            BitOr::bitor(rhs.clone(), self)
        } else {
            BitOr::bitor(self.clone(), rhs)
        }
    }
}

impl BitOrAssign<RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to an `union` between both maps.
    fn bitor_assign(&mut self, mut rhs: RoaringBitmap) {
        // We make sure that we apply the union operation on the biggest map.
        if self.len() < rhs.len() {
            mem::swap(self, &mut rhs);
        }

        for container in rhs.containers {
            let key = container.key;
            match self.containers.binary_search_by_key(&key, |c| c.key) {
                Err(loc) => self.containers.insert(loc, container),
                Ok(loc) => BitOrAssign::bitor_assign(&mut self.containers[loc], container),
            }
        }
    }
}

impl BitOrAssign<&RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to an `union` between both maps.
    fn bitor_assign(&mut self, rhs: &RoaringBitmap) {
        for container in &rhs.containers {
            let key = container.key;
            match self.containers.binary_search_by_key(&key, |c| c.key) {
                Err(loc) => self.containers.insert(loc, container.clone()),
                Ok(loc) => BitOrAssign::bitor_assign(&mut self.containers[loc], container),
            }
        }
    }
}

impl BitAnd<RoaringBitmap> for RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `intersection` between both maps.
    fn bitand(mut self, mut rhs: RoaringBitmap) -> RoaringBitmap {
        if self.len() <= rhs.len() {
            self.intersect_with(&rhs);
            self
        } else {
            rhs.intersect_with(&self);
            rhs
        }
    }
}

impl BitAnd<&RoaringBitmap> for RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `intersection` between both maps.
    fn bitand(mut self, rhs: &RoaringBitmap) -> RoaringBitmap {
        self.intersect_with(rhs);
        self
    }
}

impl BitAnd<RoaringBitmap> for &RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `intersection` between both maps.
    fn bitand(self, rhs: RoaringBitmap) -> RoaringBitmap {
        rhs & self
    }
}

impl BitAnd<&RoaringBitmap> for &RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to an `intersection` between both maps.
    fn bitand(self, rhs: &RoaringBitmap) -> RoaringBitmap {
        if self.len() <= rhs.len() {
            self.clone() & rhs
        } else {
            rhs.clone() & self
        }
    }
}

impl BitAndAssign<RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to an `intersection` between both maps.
    fn bitand_assign(&mut self, mut rhs: RoaringBitmap) {
        if self.len() <= rhs.len() {
            self.intersect_with(&rhs);
        } else {
            rhs.intersect_with(self);
            *self = rhs;
        }
    }
}

impl BitAndAssign<&RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to an `intersection` between both maps.
    fn bitand_assign(&mut self, rhs: &RoaringBitmap) {
        self.intersect_with(rhs)
    }
}

impl Sub<RoaringBitmap> for RoaringBitmap {
    type Output = crate::RoaringBitmap;

    /// This is equivalent to a `difference` between both maps.
    fn sub(mut self, rhs: RoaringBitmap) -> RoaringBitmap {
        self.difference_with(&rhs);
        self
    }
}

impl Sub<&RoaringBitmap> for RoaringBitmap {
    type Output = crate::RoaringBitmap;

    /// This is equivalent to a `difference` between both maps.
    fn sub(mut self, rhs: &RoaringBitmap) -> RoaringBitmap {
        self.difference_with(rhs);
        self
    }
}

impl Sub<RoaringBitmap> for &RoaringBitmap {
    type Output = crate::RoaringBitmap;

    /// This is equivalent to a `difference` between both maps.
    fn sub(self, rhs: RoaringBitmap) -> RoaringBitmap {
        self.clone() - rhs
    }
}

impl Sub<&RoaringBitmap> for &RoaringBitmap {
    type Output = crate::RoaringBitmap;

    /// This is equivalent to a `difference` between both maps.
    fn sub(self, rhs: &RoaringBitmap) -> RoaringBitmap {
        self.clone() - rhs
    }
}

impl SubAssign<RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to a `difference` between both maps.
    fn sub_assign(&mut self, rhs: RoaringBitmap) {
        self.difference_with(&rhs)
    }
}

impl SubAssign<&RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to a `difference` between both maps.
    fn sub_assign(&mut self, rhs: &RoaringBitmap) {
        self.difference_with(rhs)
    }
}

impl BitXor<RoaringBitmap> for RoaringBitmap {
    type Output = crate::RoaringBitmap;

    /// This is equivalent to a `symmetric difference` between both maps.
    fn bitxor(mut self, rhs: RoaringBitmap) -> RoaringBitmap {
        self.symmetric_difference_with(&rhs);
        self
    }
}

impl BitXor<&RoaringBitmap> for RoaringBitmap {
    type Output = crate::RoaringBitmap;

    /// This is equivalent to a `symmetric difference` between both maps.
    fn bitxor(mut self, rhs: &RoaringBitmap) -> RoaringBitmap {
        self.symmetric_difference_with(rhs);
        self
    }
}

impl BitXor<RoaringBitmap> for &RoaringBitmap {
    type Output = crate::RoaringBitmap;

    /// This is equivalent to a `symmetric difference` between both maps.
    fn bitxor(self, rhs: RoaringBitmap) -> RoaringBitmap {
        rhs ^ self
    }
}

impl BitXor<&RoaringBitmap> for &RoaringBitmap {
    type Output = RoaringBitmap;

    /// This is equivalent to a `symmetric difference` between both maps.
    fn bitxor(self, rhs: &RoaringBitmap) -> RoaringBitmap {
        self.clone() ^ rhs
    }
}

impl BitXorAssign<RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to a `symmetric difference` between both maps.
    fn bitxor_assign(&mut self, rhs: RoaringBitmap) {
        self.symmetric_difference_with(&rhs)
    }
}

impl BitXorAssign<&RoaringBitmap> for RoaringBitmap {
    /// This is equivalent to a `symmetric difference` between both maps.
    fn bitxor_assign(&mut self, rhs: &RoaringBitmap) {
        self.symmetric_difference_with(rhs)
    }
}
