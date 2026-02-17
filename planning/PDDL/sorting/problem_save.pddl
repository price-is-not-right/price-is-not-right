(define (problem sorting_problem)
  (:domain sorting)
  (:objects 
    milk bread cereal can - object
    box1 box2 box3 box4 - box
  )
  (:init 
    (free-gripper)
    (clear milk)
    (clear bread)
    (clear cereal)
    (clear can)
    (clear box1)
    (clear box2)
    (clear box3)
    (clear box4)
    (matches milk box1)
    (matches bread box2)
    (matches cereal box3)
    (matches can box4)
  )
  (:goal 
    (and (on milk box1)
         (on bread box2)
         (on cereal box3)
         (on can box4))
  )
)